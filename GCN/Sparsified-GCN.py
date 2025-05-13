#%% Imports and Setup
import os
import itertools
import collections

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Load sparsified graph
def load_sparsified_graph():
    # Load the saved sparsified graph data
    checkpoint = torch.load("sparsified_graph.pt")
    edge_index = checkpoint['edge_index']
    x = checkpoint['x']
    y = checkpoint['y']
    node_feature_names = checkpoint['node_feature_names']
    edge_importance = checkpoint['edge_importance']
    
    return edge_index, x, y, node_feature_names, edge_importance

#%% Built-in GNN Wrapper
class BuiltinGNN(nn.Module):
    def __init__(self,
                 model_name: str,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: int,
                 dropout: float = 0.3,
                 act: str = 'relu',
                 jk: str  = None,
                 **kwargs):
        super().__init__()
        model_map = {
            'GCN':       GCN,
            'GraphSAGE': GraphSAGE,
            'GIN':       GIN,
            'GAT':       GAT,
            'PNA':       PNA,
        }
        if model_name not in model_map:
            raise ValueError(f"Unsupported model: {model_name}")
        Wrapper = model_map[model_name]
        self.gnn = Wrapper(in_channels=in_channels,
                           hidden_channels=hidden_channels,
                           num_layers=num_layers,
                           out_channels=hidden_channels,
                           dropout=dropout,
                           act=act,
                           jk=jk,
                           **kwargs)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

#%% Training Function
def train_model(target_idx, target_name,
                model_name='GCN',
                hidden_channels=64,
                num_layers=3,
                dropout=0.3,
                jk=None,
                **kwargs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_trues = [], []

    for fold, (tr, va) in enumerate(kf.split(graphs), 1):
        tr_loader = DataLoader([graphs[i] for i in tr], batch_size=16, shuffle=True)
        va_loader = DataLoader([graphs[i] for i in va], batch_size=16)

        model = BuiltinGNN(
            model_name=model_name,
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=1,
            dropout=dropout,
            act='relu',
            jk=jk,
            **kwargs
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        # Training loop with validation
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0

        for epoch in range(1, 201):
            # Training
            model.train()
            total_loss = 0
            for batch in tr_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                target = batch.y[:, target_idx:target_idx+1]
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in va_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    target = batch.y[:, target_idx:target_idx+1]
                    val_loss += loss_fn(pred, target).item() * batch.num_graphs

            val_loss /= len(va_loader.dataset)

            if epoch % 50 == 0:
                print(f"Fold {fold} Epoch {epoch} Train MSE: {total_loss/len(tr_loader.dataset):.4f} Val MSE: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        model.load_state_dict(best_model_state)

        # Get predictions
        model.eval()
        fold_preds = []
        fold_trues = []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y[:, target_idx:target_idx+1]
                fold_preds.append(pred.cpu().numpy())
                fold_trues.append(target.cpu().numpy())

        all_preds.append(np.vstack(fold_preds))
        all_trues.append(np.vstack(fold_trues))
        print(f"Fold {fold} completed. Validation samples: {fold_preds[0].shape[0]}")

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    mse = mean_squared_error(all_trues, all_preds)
    r2 = r2_score(all_trues, all_preds)
    print(f"\n{target_name} [{model_name}] CV MSE: {mse:.4f}, R^2: {r2:.3f}")

    # Plot results
    plt.figure(figsize=(6,5))
    plt.scatter(all_trues, all_preds, alpha=0.7)
    mn, mx = all_trues.min(), all_trues.max()
    plt.plot([mn,mx], [mn,mx], 'r--')
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"{target_name} [{model_name}] Pred vs Actual")
    plt.tight_layout()
    plt.show()

    return all_preds, all_trues, mse, r2

#%% Main execution
if __name__ == "__main__":
    print("Loading sparsified graph...")
    edge_index, x, y, node_feature_names, edge_importance = load_sparsified_graph()
    
    # Create graph objects
    graphs = []
    for idx in range(len(x)):
        node_features = torch.tensor(x[idx], dtype=torch.float).unsqueeze(1)
        target = torch.tensor(y[idx], dtype=torch.float).view(1, 2)
        graph = Data(x=node_features, edge_index=edge_index, y=target)
        graphs.append(graph)
    
    print(f"Created {len(graphs)} graphs with {len(node_feature_names)} nodes each")
    print(f"Using sparsified graph with {edge_index.shape[1]//2} undirected edges")

    # PNA specific parameters
    aggregators = ['mean', 'min', 'max', 'std']
    scalers = ['identity', 'amplification', 'attenuation']
    
    # Calculate degree statistics for PNA
    from torch_geometric.utils import degree
    deg = torch.zeros(len(node_feature_names), dtype=torch.float)
    for data in graphs:
        deg += degree(data.edge_index[1], num_nodes=len(node_feature_names)).float()
    deg = deg / len(graphs)  # Average degree

    print("\n=== Training ACE-km ===")
    ace_preds, ace_trues, ace_mse, ace_r2 = train_model(
        target_idx=0,
        target_name="ACE-km",
        model_name='PNA',
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        jk='cat',
        aggregators=aggregators,
        scalers=scalers,
        deg=deg
    )

    print("\n=== Training H2-km ===")
    h2_preds, h2_trues, h2_mse, h2_r2 = train_model(
        target_idx=1,
        target_name="H2-km",
        model_name='PNA',
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        jk='cat',
        aggregators=aggregators,
        scalers=scalers,
        deg=deg
    )
