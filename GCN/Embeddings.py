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

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Data Loading and Preprocessing
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)
df = df[~df.isin(['x']).any(axis=1)]

otu_cols  = [c for c in df.columns if "d__" in c]
meta_cols = [c for c in df.columns if c not in otu_cols]

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

param_df = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

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
print(f"Selected {len(selected_families)} families after filtering.")

#%% Construct Node Features & Targets
target_cols     = ['ACE-km', 'H2-km']
target_data     = param_df[target_cols].copy()
df_microbe      = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))
node_feature_names = list(df_microbe.columns)
num_nodes       = len(node_feature_names)
feature_matrix  = df_microbe.values.astype(np.float32)
print(f"Node feature matrix: {feature_matrix.shape} (samples × nodes)")

#%% Build Edges via Mantel Test
os.makedirs('intermediate_results', exist_ok=True)

def compute_distance_matrix(vec, metric='braycurtis'):
    dm = squareform(pdist(vec[:, None], metric=metric))
    return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)

def mantel_test(d1, d2, permutations=999):
    n   = d1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1, v2 = d1[idx], d2[idx]
    if v1.std()==0 or v2.std()==0:
        return 1.0, 0.0
    r_obs = pearsonr(v1, v2)[0]
    count = sum(
        abs(pearsonr(v1, np.random.permutation(v2))[0]) >= abs(r_obs)
        for _ in range(permutations)
    )
    return (count + 1) / (permutations + 1), r_obs

dist_mats = {
    fam: compute_distance_matrix(feature_matrix[:, i], metric='braycurtis')
    for i, fam in enumerate(node_feature_names)
}

edge_i, edge_j = [], []
for i, j in itertools.combinations(range(num_nodes), 2):
    p, _ = mantel_test(dist_mats[node_feature_names[i]],
                       dist_mats[node_feature_names[j]], permutations=999)
    if p < 0.05:
        edge_i += [i, j]
        edge_j += [j, i]

edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
print("Total undirected edges:", edge_index.shape[1] // 2)

#%% Create Graph Data Objects
graphs = []
for idx in range(len(feature_matrix)):
    x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)
    y = torch.tensor(target_data.iloc[idx].values, dtype=torch.float).view(1, 2)
    graphs.append(Data(x=x, edge_index=edge_index, y=y))
print(f"Created {len(graphs)} graphs with {num_nodes} nodes each.")

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
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

    def get_graph_embeddings(self, data):
        x = self.gnn(data.x, data.edge_index, data.batch)
        x = global_mean_pool(x, data.batch)
        return x  # [batch_size, hidden_channels]

#%% Train GNN and Extract Embeddings
def train_and_evaluate_pipeline(target_idx, model_name='GIN', hidden_channels=64, num_layers=3, dropout=0.3, jk=None, **kwargs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize metrics storage
    gnn_val_scores = []
    svr_val_scores = []
    et_val_scores = []
    
    for fold, (tr, va) in enumerate(kf.split(graphs), 1):
        print(f"\nProcessing fold {fold}/5...")
        
        # Split data
        tr_loader = DataLoader([graphs[i] for i in tr], batch_size=16, shuffle=True)
        va_loader = DataLoader([graphs[i] for i in va], batch_size=16)
        
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
        gnn_val_scores.append(best_val_loss)
        
        # Generate embeddings for training and validation sets
        model.eval()
        tr_embeddings = []
        va_embeddings = []
        tr_targets = []
        va_targets = []
        
        with torch.no_grad():
            # Training set embeddings
            for batch in tr_loader:
                batch = batch.to(device)
                emb = model.get_graph_embeddings(batch)
                tr_embeddings.append(emb.cpu().numpy())
                tr_targets.append(batch.y[:, target_idx:target_idx+1].cpu().numpy())
            
            # Validation set embeddings
            for batch in va_loader:
                batch = batch.to(device)
                emb = model.get_graph_embeddings(batch)
                va_embeddings.append(emb.cpu().numpy())
                va_targets.append(batch.y[:, target_idx:target_idx+1].cpu().numpy())
        
        tr_embeddings = np.vstack(tr_embeddings)
        va_embeddings = np.vstack(va_embeddings)
        tr_targets = np.vstack(tr_targets).ravel()
        va_targets = np.vstack(va_targets).ravel()
        
        # Train and evaluate LinearSVR
        svr = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
        svr.fit(tr_embeddings, tr_targets)
        svr_pred = svr.predict(va_embeddings)
        svr_mse = mean_squared_error(va_targets, svr_pred)
        svr_val_scores.append(svr_mse)
        
        # Train and evaluate ExtraTrees
        et = ExtraTreesRegressor(n_estimators=100, random_state=42)
        et.fit(tr_embeddings, tr_targets)
        et_pred = et.predict(va_embeddings)
        et_mse = mean_squared_error(va_targets, et_pred)
        et_val_scores.append(et_mse)
        
        print(f"Fold {fold} Results:")
        print(f"GNN Val MSE: {best_val_loss:.4f}")
        print(f"LinearSVR Val MSE: {svr_mse:.4f}")
        print(f"ExtraTrees Val MSE: {et_mse:.4f}")
    
    # Print final results
    print("\nFinal Results:")
    print(f"GNN Average Val MSE: {np.mean(gnn_val_scores):.4f} ± {np.std(gnn_val_scores):.4f}")
    print(f"LinearSVR Average Val MSE: {np.mean(svr_val_scores):.4f} ± {np.std(svr_val_scores):.4f}")
    print(f"ExtraTrees Average Val MSE: {np.mean(et_val_scores):.4f} ± {np.std(et_val_scores):.4f}")

#%% Main: Run the complete pipeline
if __name__ == "__main__":
    # Choose which target to predict (0: ACE-km, 1: H2-km)
    target_idx = 0
    train_and_evaluate_pipeline(
        target_idx=target_idx,
        model_name='GIN',
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        jk='cat'
    )
# %%
