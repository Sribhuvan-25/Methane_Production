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

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import networkx as nx

# Load sparsified graph
sparsified_graph_path = 'Sparsified_Graphs/sparsified_graph_k5.pt'
if not os.path.exists(sparsified_graph_path):
    raise FileNotFoundError(f"Sparsified graph not found at {sparsified_graph_path}. Please run Sparsify.py first.")

print('Loading sparsified graph...')
sg = torch.load(sparsified_graph_path, weights_only=False)
edge_index = sg['edge_index']
edge_attr = sg.get('edge_attr', None)
print(f"Loaded sparsified graph with {edge_index.shape[1]//2} undirected edges.")
if edge_attr is not None:
    print(f"Edge weights available with mean: {edge_attr.mean().item():.4f}")

# Debug: print edge_index and edge_attr shapes
print(f"edge_index shape: {edge_index.shape}")
if edge_attr is not None:
    print(f"edge_attr shape: {edge_attr.shape}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Data Loading and Preprocessing
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)
print(f"Loaded df shape: {df.shape}")
df = df[~df.isin(['x']).any(axis=1)]
print(f"Filtered df shape (no 'x'): {df.shape}")

otu_cols  = [c for c in df.columns if "d__" in c]
meta_cols = [c for c in df.columns if c not in otu_cols]

print(f"Number of OTU columns: {len(otu_cols)}")
print(f"Number of meta columns: {len(meta_cols)}")

def extract_family_from_colname(colname):
    for part in colname.split(';'):
        part = part.strip()
        if part.startswith('f__'):
            return part[3:] or "UnclassifiedFamily"
    return "UnclassifiedFamily"

col_to_family = {c: extract_family_from_colname(c) for c in otu_cols}
family_to_cols = collections.defaultdict(list)
for c, fam in col_to_family.items():
    family_to_cols[fam].append(c)

df_fam = pd.DataFrame({
    fam: df[cols].sum(axis=1)
    for fam, cols in family_to_cols.items()
}, index=df.index)
print(f"df_fam shape: {df_fam.shape}")

param_df = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
print(f"df_fam_rel shape: {df_fam_rel.shape}")

# Save for later
df_fam_rel.to_csv("family_abundance.csv", index=False)
param_df.to_csv("parameters.csv", index=False)

#%% Filtering Families
df_fam_rel = pd.read_csv("family_abundance.csv")
param_df   = pd.read_csv("parameters.csv")
param_df.columns = param_df.columns.str.strip()

presence_count = (df_fam_rel > 0).sum(0)
prevalence     = presence_count / df_fam_rel.shape[0]
high_prev      = prevalence[prevalence >= 0.05].index

mean_abund     = df_fam_rel.mean(0)
high_abund     = mean_abund[mean_abund >= 0.01].index

selected_families = high_prev.intersection(high_abund)
df_fam_rel_filtered = df_fam_rel[selected_families].copy()
print(f"Selected {len(selected_families)} families after filtering: {list(selected_families)}")
print(f"df_fam_rel_filtered shape: {df_fam_rel_filtered.shape}")

#%% Construct Node Features & Targets
target_cols     = ['ACE-km', 'H2-km']
target_data     = param_df[target_cols].copy()
df_microbe      = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))
node_feature_names = list(df_microbe.columns)
num_nodes       = len(node_feature_names)
feature_matrix  = df_microbe.values.astype(np.float32)
print(f"Node feature matrix: {feature_matrix.shape} (samples × nodes)")
print(f"Target data shape: {target_data.shape}")

#%% Create Graph Data Objects
graphs = []
for idx in range(len(feature_matrix)):
    x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)
    y = torch.tensor(target_data.iloc[idx].values, dtype=torch.float).view(1, 2)
    # Ensure edge_index and edge_attr are on the same device as x
    edge_index_device = edge_index.to(x.device)
    edge_attr_device = edge_attr.to(x.device) if edge_attr is not None else None
    graphs.append(Data(x=x, edge_index=edge_index_device, edge_attr=edge_attr_device, y=y))
print(f"Created {len(graphs)} graphs with {num_nodes} nodes each.")

# Debug: print a sample graph's details
sample_idx = 0
sample_graph = graphs[sample_idx]
print(f"Sample graph {sample_idx} x shape: {sample_graph.x.shape}")
print(f"Sample graph {sample_idx} edge_index shape: {sample_graph.edge_index.shape}")
if sample_graph.edge_attr is not None:
    print(f"Sample graph {sample_idx} edge_attr shape: {sample_graph.edge_attr.shape}")

# Visualize the sample graph
print("Visualizing a sample graph (first sample)...")
G_nx = nx.Graph()
edge_index_np = sample_graph.edge_index.cpu().numpy()
G_nx.add_edges_from(zip(edge_index_np[0], edge_index_np[1]))
labels = {i: name for i, name in enumerate(node_feature_names)}
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G_nx, seed=42)
nx.draw(G_nx, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=8, font_weight='bold', edge_color='#cccccc')
plt.title('Sample Graph Visualization')
plt.show()

#%% Built-in GNN Wrapper with Embedding Extraction
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
        # C. Fix head dimension for jk='cat'
        head_in_dim = hidden_channels * num_layers if jk == 'cat' else hidden_channels
        self.head = nn.Linear(head_in_dim, out_channels)

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

    def get_graph_embeddings(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        x = global_mean_pool(x, data.batch)
        return x  # [batch_size, hidden_channels]

#%% Train GNN and Extract Embeddings
# B. Add train/val/test split, remove early stopping, visualize embeddings, add raw feature baseline

def train_and_evaluate_pipeline(target_idx, target_name, model_name='GCN', hidden_channels=64, num_layers=3, dropout=0.3, jk=None, **kwargs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_val_preds_svr, all_val_trues, all_val_preds_et = [], [], []
    all_val_embs, all_val_targets = [], []
    all_val_indices = []
    all_val_preds_svr_raw, all_val_preds_et_raw = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(graphs), 1):
        print(f"\nProcessing fold {fold}/5...")
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=16)

        # Train GNN
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
        num_epochs = 200
        best_val_loss = float('inf')
        best_model_state = None
        for epoch in range(1, num_epochs+1):
            model.train()
            total_loss = 0
            for batch in train_loader:
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
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    target = batch.y[:, target_idx:target_idx+1]
                    val_loss += loss_fn(pred, target).item() * batch.num_graphs
            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            if epoch % 50 == 0:
                print(f"Epoch {epoch} Train MSE: {total_loss/len(train_loader.dataset):.4f} Val MSE: {val_loss:.4f}")
        model.load_state_dict(best_model_state)

        # Get embeddings and targets for validation set
        model.eval()
        val_embs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                emb = model.get_graph_embeddings(batch)
                val_embs.append(emb.cpu().numpy())
                val_targets.append(batch.y[:, target_idx:target_idx+1].cpu().numpy())
        val_embs = np.vstack(val_embs)
        val_targets = np.vstack(val_targets).ravel()
        all_val_embs.append(val_embs)
        all_val_targets.append(val_targets)
        all_val_indices.extend(val_idx)

        # Downstream ML on GNN embeddings (fit on train, predict on val)
        # Get train embeddings/targets
        train_embs, train_targets = [], []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                emb = model.get_graph_embeddings(batch)
                train_embs.append(emb.cpu().numpy())
                train_targets.append(batch.y[:, target_idx:target_idx+1].cpu().numpy())
        train_embs = np.vstack(train_embs)
        train_targets = np.vstack(train_targets).ravel()
        # SCALE GNN EMBEDDINGS
        scaler_emb = StandardScaler()
        train_embs_scaled = scaler_emb.fit_transform(train_embs)
        val_embs_scaled = scaler_emb.transform(val_embs)
        # SVR
        svr = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
        svr.fit(train_embs_scaled, train_targets)
        svr_pred = svr.predict(val_embs_scaled)
        all_val_preds_svr.append(svr_pred)
        # ExtraTrees
        et = ExtraTreesRegressor(n_estimators=100, random_state=42)
        et.fit(train_embs, train_targets)
        et_pred = et.predict(val_embs)
        all_val_preds_et.append(et_pred)

        # D. Baseline: Downstream ML on raw features (no GNN)
        raw_features = feature_matrix # shape: (num_samples, num_nodes)
        raw_targets = target_data.values[:, target_idx]
        X_train = raw_features[train_idx]
        X_val = raw_features[val_idx]
        y_train = raw_targets[train_idx]
        y_val = raw_targets[val_idx]
        # SCALE RAW FEATURES
        scaler_raw = StandardScaler()
        X_train_scaled = scaler_raw.fit_transform(X_train)
        X_val_scaled = scaler_raw.transform(X_val)
        # LinearSVR
        svr_raw = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
        svr_raw.fit(X_train_scaled, y_train)
        svr_raw_pred = svr_raw.predict(X_val_scaled)
        all_val_preds_svr_raw.append(svr_raw_pred)
        # ExtraTrees
        et_raw = ExtraTreesRegressor(n_estimators=100, random_state=42)
        et_raw.fit(X_train, y_train)
        et_raw_pred = et_raw.predict(X_val)
        all_val_preds_et_raw.append(et_raw_pred)

    # After all folds: concatenate
    all_val_embs = np.vstack(all_val_embs)
    all_val_targets = np.concatenate(all_val_targets)
    all_val_preds_svr = np.concatenate(all_val_preds_svr)
    all_val_preds_et = np.concatenate(all_val_preds_et)
    all_val_preds_svr_raw = np.concatenate(all_val_preds_svr_raw)
    all_val_preds_et_raw = np.concatenate(all_val_preds_et_raw)

    # C. Embedding visualization (PCA/t-SNE) on all validation embeddings
    pca = PCA(n_components=2)
    pca_embs = pca.fit_transform(all_val_embs)
    plt.figure(figsize=(8,6))
    plt.scatter(pca_embs[:,0], pca_embs[:,1], c=all_val_targets, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Target')
    plt.title(f'PCA of GNN Embeddings (val folds) for {target_name}')
    plt.show()

    # Metrics for GNN embeddings
    svr_mse = mean_squared_error(all_val_targets, all_val_preds_svr)
    svr_r2 = r2_score(all_val_targets, all_val_preds_svr)
    et_mse = mean_squared_error(all_val_targets, all_val_preds_et)
    et_r2 = r2_score(all_val_targets, all_val_preds_et)
    print(f"\n5-Fold CV Results for {target_name} (GNN Embeddings):")
    print(f"LinearSVR MSE: {svr_mse:.4f}, R²: {svr_r2:.4f}")
    print(f"ExtraTrees MSE: {et_mse:.4f}, R²: {et_r2:.4f}")
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(all_val_targets, all_val_preds_svr, alpha=0.7)
    mn, mx = all_val_targets.min(), all_val_targets.max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"LinearSVR: {target_name} Pred vs Actual\nMSE: {svr_mse:.4f}, R²: {svr_r2:.4f}")
    plt.subplot(1, 2, 2)
    plt.scatter(all_val_targets, all_val_preds_et, alpha=0.7)
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"ExtraTrees: {target_name} Pred vs Actual\nMSE: {et_mse:.4f}, R²: {et_r2:.4f}")
    plt.tight_layout()
    plt.show()

    # Baseline: raw features
    svr_raw_mse = mean_squared_error(all_val_targets, all_val_preds_svr_raw)
    svr_raw_r2 = r2_score(all_val_targets, all_val_preds_svr_raw)
    et_raw_mse = mean_squared_error(all_val_targets, all_val_preds_et_raw)
    et_raw_r2 = r2_score(all_val_targets, all_val_preds_et_raw)
    print(f"LinearSVR (raw) MSE: {svr_raw_mse:.4f}, R²: {svr_raw_r2:.4f}")
    print(f"ExtraTrees (raw) MSE: {et_raw_mse:.4f}, R²: {et_raw_r2:.4f}")
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(all_val_targets, all_val_preds_svr_raw, alpha=0.7)
    mn, mx = all_val_targets.min(), all_val_targets.max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"LinearSVR (raw): {target_name} Pred vs Actual\nMSE: {svr_raw_mse:.4f}, R²: {svr_raw_r2:.4f}")
    plt.subplot(1, 2, 2)
    plt.scatter(all_val_targets, all_val_preds_et_raw, alpha=0.7)
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"ExtraTrees (raw): {target_name} Pred vs Actual\nMSE: {et_raw_mse:.4f}, R²: {et_raw_r2:.4f}")
    plt.tight_layout()
    plt.show()

#%% Main: Run the complete pipeline for both targets
if __name__ == "__main__":
    # PNA specific parameters
    aggregators = ['mean', 'min', 'max', 'std']
    scalers = ['identity', 'amplification', 'attenuation']
    from torch_geometric.utils import degree
    deg = torch.zeros(num_nodes, dtype=torch.float)
    for data in graphs:
        deg += degree(data.edge_index[1], num_nodes=num_nodes).float()
    deg = deg / len(graphs)
    print("=== Training ACE-km ===")
    train_and_evaluate_pipeline(
        target_idx=0,
        target_name="ACE-km",
        model_name='GCN',
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        jk=None,
        # heads=4,
    )
    
    print("=== Training H2-km ===")
    train_and_evaluate_pipeline(
        target_idx=1,
        target_name="H2-km",
        model_name='GCN',
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        jk=None,
        # heads=4,
    )


# %%
