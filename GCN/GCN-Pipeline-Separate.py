#%% Imports and Setup
import pandas as pd
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import itertools
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Data Loading and Preprocessing
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)
df = df[~df.isin(['x']).any(axis=1)]
print(df.columns)
otu_cols = [col for col in df.columns if "d__" in col]
meta_cols = [col for col in df.columns if col not in otu_cols]

def extract_family_from_colname(colname):
    parts = colname.split(';')
    for part in parts:
        part = part.strip()
        if part.startswith('f__'):
            fam = part[3:]
            return fam if fam else "UnclassifiedFamily"
    return "UnclassifiedFamily"

col_to_family = {}
for c in otu_cols:
    family = extract_family_from_colname(c)
    col_to_family[c] = family

family_to_cols = collections.defaultdict(list)
for c in otu_cols:
    fam = col_to_family[c]
    family_to_cols[fam].append(c)

df_fam = pd.DataFrame(index=df.index)
for fam, col_list in family_to_cols.items():
    df_fam[fam] = df[col_list].sum(axis=1)

param_df = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

df_fam_rel.to_csv("family_abundance.csv", index=False)
param_df.to_csv("parameters.csv", index=False)

#%% Filtering Families
df_fam_rel = pd.read_csv("family_abundance.csv")
param_df = pd.read_csv("parameters.csv")
param_df.columns = param_df.columns.str.strip()

presence_count = (df_fam_rel > 0).sum(axis=0)
prevalence = presence_count / df_fam_rel.shape[0]
prev_threshold = 0.05
high_prev_families = prevalence[prevalence >= prev_threshold].index

mean_abund = df_fam_rel.mean(axis=0)
abund_threshold = 0.01
high_abund_families = mean_abund[mean_abund >= abund_threshold].index

selected_families = high_prev_families.intersection(high_abund_families)
df_fam_rel_filtered = df_fam_rel[selected_families].copy()
print(f"Selected {len(selected_families)} families after filtering.")

#%% Constructing Node Features
target_cols = ['ACE-km', 'H2-km']

# Extract targets (graph-level labels)
target_data = param_df[target_cols].copy()

# Remove the target columns from param_inputs if they exist
param_inputs = param_df.drop(columns=target_cols, errors='ignore')

# Double square-root transformation for microbial relative abundances
df_microbe_sqrt = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))

# ONLY use microbial features as nodes
node_features_df = df_microbe_sqrt.copy()
print("Node feature matrix shape (samples x nodes):", node_features_df.shape)

# Record feature names - only microbial features now
node_feature_names = list(node_features_df.columns)
print(f"Number of node features: {len(node_feature_names)}")

#%% Edge Construction via Mantel Test
def compute_distance_matrix(vec, metric='euclidean'):
    if metric == 'euclidean':
        dm = squareform(pdist(vec[:, None], metric='euclidean'))
    elif metric == 'braycurtis':
        dm = squareform(pdist(vec[:, None], metric='braycurtis'))
    else:
        raise ValueError("Unsupported metric.")
    dm = np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    return dm

os.makedirs('intermediate_results', exist_ok=True)

demo_nodes = node_feature_names[:2]
print("\n=== DEMONSTRATION OF DISTANCE MATRIX COMPUTATION AND MANTEL TEST ===")
print(f"Selected nodes for demonstration: {demo_nodes}")

f = open('intermediate_results/distance_matrices_and_edges.txt', 'w')
f.write("=== DISTANCE MATRICES AND EDGE ESTABLISHMENT ANALYSIS ===\n\n")

dist_mats = {}
for col in node_feature_names:
    values = node_features_df[col].values.astype(np.float32)
    if col in demo_nodes:
        f.write(f"\n--- Processing node: {col} ---\n")
        f.write(f"Raw values (first 10): {values[:10]}\n")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        f.write(f"Values after NaN/inf replacement (first 10): {values[:10]}\n")
        dist_mats[col] = compute_distance_matrix(values, metric='braycurtis')
        f.write(f"Distance matrix shape: {dist_mats[col].shape}\n")
        f.write(f"Distance matrix (first 10x10):\n")
        np.savetxt(f, dist_mats[col][:10, :10], fmt='%.4f')
        f.write("\n")
    else:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        dist_mats[col] = compute_distance_matrix(values, metric='braycurtis')

def mantel_test(dist_matrix1, dist_matrix2, permutations=999):
    n = dist_matrix1.shape[0]
    idx = np.triu_indices(n, k=1)
    d1 = dist_matrix1[idx]
    d2 = dist_matrix2[idx]
    
    f.write(f"\n--- Mantel Test Details ---\n")
    f.write(f"Upper triangular indices shape: {idx[0].shape}\n")
    f.write(f"Extracted distance vectors shape: d1={d1.shape}, d2={d2.shape}\n")
    f.write(f"Distance vector means: d1={d1.mean():.4f}, d2={d2.mean():.4f}\n")
    
    if d1.std() == 0 or d2.std() == 0:
        f.write("Warning: Zero standard deviation detected\n")
        return 1.0, 0.0
    
    r_obs = pearsonr(d1, d2)[0]
    f.write(f"Observed correlation (r_obs): {r_obs:.4f}\n")
    
    count = 0
    for i in range(permutations):
        perm = np.random.permutation(n)
        d2_perm = dist_matrix2[perm][:, perm]
        r_perm = pearsonr(d1, d2_perm[idx])[0]
        if abs(r_perm) >= abs(r_obs):
            count += 1
        if i < 5:
            f.write(f"Permutation {i+1}: r_perm={r_perm:.4f}\n")
    
    p_val = (count + 1) / (permutations + 1)
    f.write(f"Final p-value: {p_val:.4f}\n")
    return p_val, r_obs

f.write("\n=== MANTEL TEST RESULTS BETWEEN DEMO NODES ===\n")
for i, j in itertools.combinations(range(len(demo_nodes)), 2):
    col_i = demo_nodes[i]
    col_j = demo_nodes[j]
    f.write(f"\nTesting edge between: {col_i} and {col_j}\n")
    p_val, r_val = mantel_test(dist_mats[col_i], dist_mats[col_j], permutations=999)
    f.write(f"Results: p-value={p_val:.4f}, correlation={r_val:.4f}\n")
    if p_val < 0.05:
        f.write("SIGNIFICANT EDGE FOUND!\n")
    else:
        f.write("No significant edge\n")

edge_index_list = [[], []]
num_nodes = len(node_feature_names)
for i, j in itertools.combinations(range(num_nodes), 2):
    col_i = node_feature_names[i]
    col_j = node_feature_names[j]
    p_val, r_val = mantel_test(dist_mats[col_i], dist_mats[col_j], permutations=999)
    if p_val < 0.05:
        edge_index_list[0].extend([i, j])
        edge_index_list[1].extend([j, i])
edge_index = torch.tensor(edge_index_list, dtype=torch.long)
print("\nTotal edges (undirected) defined by Mantel test:", edge_index.shape[1]//2)

f.close()
print("Intermediate results saved to 'intermediate_results/distance_matrices_and_edges.txt'")

#%% Create PyG Data Objects for Each Sample
node_features_df = node_features_df.apply(pd.to_numeric, errors='coerce')
print("Node feature matrix shape (samples x nodes):", node_features_df.shape)
node_features = node_features_df.values.astype(np.float32)

# Graph-level target values
target_data = target_data.apply(pd.to_numeric, errors='coerce')

graphs = []
for idx in range(len(node_features_df)):
    # Create node features tensor
    x = torch.tensor(node_features[idx], dtype=torch.float).unsqueeze(1)  # shape: [num_nodes, 1]
    
    # Create graph-level target tensor
    y_vals = target_data.iloc[idx].values.astype(np.float32)
    y = torch.tensor(y_vals, dtype=torch.float).view(-1, 2)  # shape: [1, 2]
    
    # Create graph with edge_index and graph-level target
    graph = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(graph)

print(f"Created {len(graphs)} graph data objects (one per sample).")
print(f"Each graph has {num_nodes} nodes (microbial families) and {edge_index.shape[1]//2} undirected edges.")
print(f"Target dimensions: {graphs[0].y.shape}")

#%% Model Definition for Single Output

import torch
import torch.nn as nn
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA  # wrapper models
from torch_geometric.nn import global_mean_pool                             # graph readout

class BuiltinGNN(nn.Module):
    def __init__(self,
                 model_name: str,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 jk: str = None,
                 **kwargs):
        """
        model_name: one of "GCN", "GraphSAGE", "GIN", "GAT", "PNA"
        kwargs are passed to the chosen model (e.g., aggregators for PNA)
        """
        super().__init__()
        if model_name == 'GCN':
            self.gnn = GCN(in_channels,
                           hidden_channels,
                           num_layers,
                           out_channels=hidden_channels,
                           dropout=dropout,
                           act=act,
                           jk=jk,
                           **kwargs)           # :contentReference[oaicite:5]{index=5}
        elif model_name == 'GraphSAGE':
            self.gnn = GraphSAGE(in_channels,
                                 hidden_channels,
                                 num_layers,
                                 out_channels=hidden_channels,
                                 dropout=dropout,
                                 act=act,
                                 jk=jk,
                                 **kwargs)      # :contentReference[oaicite:6]{index=6}
        elif model_name == 'GIN':
            self.gnn = GIN(in_channels,
                           hidden_channels,
                           num_layers,
                           out_channels=hidden_channels,
                           dropout=dropout,
                           act=act,
                           jk=jk,
                           **kwargs)          # :contentReference[oaicite:7]{index=7}
        elif model_name == 'GAT':
            self.gnn = GAT(in_channels,
                           hidden_channels,
                           num_layers,
                           out_channels=hidden_channels,
                           dropout=dropout,
                           act=act,
                           jk=jk,
                           **kwargs)          # :contentReference[oaicite:8]{index=8}
        elif model_name == 'PNA':
            self.gnn = PNA(in_channels,
                           hidden_channels,
                           num_layers,
                           out_channels=hidden_channels,
                           dropout=dropout,
                           act=act,
                           jk=jk,
                           **kwargs)          # :contentReference[oaicite:9]{index=9}
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # final projection to your target dimension
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # data.x: [total_nodes, in_channels]
        # data.edge_index: [2, total_edges]
        # data.batch: [total_nodes] with graph indices
        x = self.gnn(data.x, data.edge_index, data.batch)  # returns [batch_size, hidden_channels]
        x = global_mean_pool(x, data.batch)                # :contentReference[oaicite:10]{index=10}
        return self.head(x)                                # [batch_size, out_channels]



class PaperGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, fc_hidden_dims=[128, 64, 32, 16], out_channels=1):
        super(PaperGCNModel, self).__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        self.fc1 = nn.Linear(hidden_channels, fc_hidden_dims[0])
        self.fc2 = nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1])
        self.fc3 = nn.Linear(fc_hidden_dims[1], fc_hidden_dims[2])
        self.fc4 = nn.Linear(fc_hidden_dims[2], fc_hidden_dims[3])
        
        self.out = nn.Linear(fc_hidden_dims[3], out_channels)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        out = self.out(x)
        return out
from torch_geometric.nn import global_add_pool, global_mean_pool, GATConv
class ImprovedGATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, heads=4, dropout=0.3):
        super(ImprovedGATModel, self).__init__()
        
        # GAT layers with multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1)
        
        # Layer normalization (more efficient than batch norm for small batches)
        self.norm1 = nn.LayerNorm(hidden_channels)
        
        # Attention-based pooling
        self.pool_attn = nn.Linear(hidden_channels, 1)
        
        # Simplified FC layers (just two instead of four)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer with attention
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU is often better than ReLU for GAT
        x = self.dropout(x)
        
        # Second GAT layer with residual connection
        x_res = x
        x = self.conv2(x, edge_index)
        x = x + x_res  # Residual connection
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Attention-based pooling (instead of simple mean pooling)
        attn_weights = self.pool_attn(x)
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Apply attention weights per graph
        num_nodes = x.size(0)
        if batch is None:
            # If there's only one graph
            x_pooled = (x * attn_weights).sum(dim=0, keepdim=True)
        else:
            # If there are multiple graphs in a batch
            x_pooled = global_add_pool(x * attn_weights, batch)
        
        # FC layers with residual connection
        x = self.fc1(x_pooled)
        x = F.elu(x)
        x = self.dropout(x)
        
        out = self.fc2(x)
        return out
#%% Training Function for Single Target
def train_model(target_idx, target_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_fold_preds = []
    all_fold_trues = []

    for train_idx, val_idx in kf.split(graphs):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
        
        # model = PaperGCNModel(
        #     in_channels=1,
        #     hidden_channels=64,
        #     fc_hidden_dims=[128, 64, 32, 16],
        #     out_channels=1
        # ).to(device)
        
        # model = ImprovedGATModel(
        #     in_channels=1,
        #     hidden_channels=64,
        #     out_channels=1,
        #     heads=4,
        #     dropout=0.6
        # ).to(device)
        
        model = BuiltinGNN(
            model_name='GCN',
            in_channels=1,
            hidden_channels=64,
            num_layers=3,
            out_channels=1,
            dropout=0.3,
            act='relu',
            jk='cat',
            # for PNA you might pass:
            # aggregators=['mean','max','sum','min'],
            # scalers=['identity','amplification','attenuation'],
            # edge_dim=...,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        
        for epoch in range(1, 301):
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
            if epoch % 20 == 0:
                print(f"{target_name} - Fold {fold} Epoch {epoch:03d} | Train Loss: {total_loss/len(train_graphs):.4f}")
        
        model.eval()
        fold_preds = []
        fold_trues = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y[:, target_idx:target_idx+1]
                fold_preds.append(pred.cpu().numpy())
                fold_trues.append(target.cpu().numpy())
        fold_preds = np.vstack(fold_preds)
        fold_trues = np.vstack(fold_trues)
        all_fold_preds.append(fold_preds)
        all_fold_trues.append(fold_trues)
        print(f"{target_name} - Fold {fold} completed. Validation samples: {fold_preds.shape[0]}")
        fold += 1

    all_preds = np.vstack(all_fold_preds)
    all_trues = np.vstack(all_fold_trues)

    mse = mean_squared_error(all_trues, all_preds)
    r2 = r2_score(all_trues, all_preds)
    print(f"\n{target_name} Results:")
    print(f"Overall CV MSE: {mse:.4f}")
    print(f"Overall CV R^2: {r2:.3f}")

    plt.figure(figsize=(6,5))
    plt.scatter(all_trues, all_preds, alpha=0.7, color='blue', edgecolors='k')
    min_val = min(all_trues.min(), all_preds.min())
    max_val = max(all_trues.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal fit')
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Predicted vs Actual: {target_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return all_preds, all_trues, mse, r2

#%% Train Separate Models for ACE-km and H2-km
print("Training ACE-km Model...")
ace_preds, ace_trues, ace_mse, ace_r2 = train_model(0, "ACE-km")

print("\nTraining H2-km Model...")
h2_preds, h2_trues, h2_mse, h2_r2 = train_model(1, "H2-km") 
# %%


# 