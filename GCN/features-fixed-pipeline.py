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
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

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

param_df   = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

df_fam_rel.to_csv("family_abundance.csv", index=False)
param_df.to_csv("parameters.csv", index=False)


#%% Filtering Families (MODIFIED)
df_fam_rel = pd.read_csv("family_abundance.csv")
param_df   = pd.read_csv("parameters.csv")
param_df.columns = param_df.columns.str.strip()

# Calculate prevalence and abundance for filtering
presence_count = (df_fam_rel > 0).sum(0)
prevalence     = presence_count / df_fam_rel.shape[0]
high_prev      = prevalence[prevalence >= 0.05].index

mean_abund     = df_fam_rel.mean(0)
high_abund     = mean_abund[mean_abund >= 0.01].index

# Find families that meet the standard filtering criteria
standard_filtered = high_prev.intersection(high_abund)

# Find families that start with "Methano" or "Syntro" (case insensitive)
methano_syntro_families = []
for fam in df_fam_rel.columns:
    fam_lower = fam.lower()
    if fam_lower.startswith('methano') or fam_lower.startswith('syntro'):
        methano_syntro_families.append(fam)

# Print the families of interest
print(f"Found {len(methano_syntro_families)} Methano/Syntro families:")
for fam in methano_syntro_families:
    print(f"  - {fam}")

# Combine both sets of families
selected_families = list(set(standard_filtered) | set(methano_syntro_families))
print(f"Selected {len(selected_families)} families in total:")
print(f"  - {len(standard_filtered)} from standard filtering")
print(f"  - {len(methano_syntro_families)} from Methano/Syntro criteria")
print(f"  - {len(selected_families) - len(standard_filtered) - len([f for f in methano_syntro_families if f not in standard_filtered])} overlap")

# Create filtered dataframe
df_fam_rel_filtered = df_fam_rel[selected_families].copy()


#%% Construct Node Features & Targets
# target_cols    = ['ACE-km', 'H2-km']
# target_data    = param_df[target_cols].copy()
# df_microbe     = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))
# node_feature_names = list(df_microbe.columns)
# num_nodes      = len(node_feature_names)
# feature_matrix = df_microbe.values.astype(np.float32)
# print(f"Node feature matrix: {feature_matrix.shape} (samples × nodes)")

# Constructing Node Features
target_cols = ['ACE-km', 'H2-km']

# Extract targets (graph-level labels)
target_data = param_df[target_cols].copy()

# Remove the target columns from param_inputs if they exist
param_inputs = param_df.drop(columns=target_cols, errors='ignore')

# Double square-root transformation for microbial relative abundances
df_microbe_sqrt = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))

# ONLY use microbial features as nodes (no parameter features)
node_features_df = df_microbe_sqrt.copy()
print("Node feature matrix shape (samples x nodes):", node_features_df.shape)

# Record feature names - only microbial features now
node_feature_names = list(node_features_df.columns)
print(f"Number of node features: {len(node_feature_names)}")


#%% Build Edges via Mantel Test
os.makedirs('intermediate_results', exist_ok=True)

# def compute_distance_matrix(vec, metric='braycurtis'):
#     dm = squareform(pdist(vec[:, None], metric=metric))
#     return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)

# def mantel_test(d1, d2, permutations=999):
#     n   = d1.shape[0]
#     idx = np.triu_indices(n, k=1)
#     v1, v2 = d1[idx], d2[idx]
#     if v1.std()==0 or v2.std()==0:
#         return 1.0, 0.0
#     r_obs = pearsonr(v1, v2)[0]
#     count = sum(
#         abs(pearsonr(v1, np.random.permutation(v2))[0]) >= abs(r_obs)
#         for _ in range(permutations)
#     )
#     return (count + 1) / (permutations + 1), r_obs

# dist_mats = {
#     fam: compute_distance_matrix(feature_matrix[:, i], metric='braycurtis')
#     for i, fam in enumerate(node_feature_names)
# }

# edge_i, edge_j = [], []
# for i, j in itertools.combinations(range(num_nodes), 2):
#     p, _ = mantel_test(dist_mats[node_feature_names[i]],
#                        dist_mats[node_feature_names[j]], permutations=999)
#     if p < 0.05:
#         edge_i += [i, j]
#         edge_j += [j, i]

# edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
# print("Total undirected edges:", edge_index.shape[1] // 2)

#%% Edge Construction via Mantel Test
def compute_distance_matrix(vec, metric='euclidean'):
    if metric == 'euclidean':
        dm = squareform(pdist(vec[:, None], metric='euclidean'))
    elif metric == 'braycurtis':
        dm = squareform(pdist(vec[:, None], metric='braycurtis'))
    else:
        raise ValueError("Unsupported metric.")
    # Replace any NaNs or infs in the computed distance matrix
    dm = np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    return dm

# Create a directory for saving intermediate results
os.makedirs('intermediate_results', exist_ok=True)

# Select 2 nodes for detailed analysis
demo_nodes = node_feature_names[:2]  # Take first 2 nodes for demonstration
print("\n=== DEMONSTRATION OF DISTANCE MATRIX COMPUTATION AND MANTEL TEST ===")
print(f"Selected nodes for demonstration: {demo_nodes}")

# Open a file to save intermediate results
f = open('intermediate_results/distance_matrices_and_edges.txt', 'w')
f.write("=== DISTANCE MATRICES AND EDGE ESTABLISHMENT ANALYSIS ===\n\n")

# Compute distance matrices for all nodes
dist_mats = {}
for col in node_feature_names:
    values = node_features_df[col].values.astype(np.float32)
    
    # If this is a demo node, save detailed information
    if col in demo_nodes:
        f.write(f"\n--- Processing node: {col} ---\n")
        f.write(f"Raw values (first 10): {values[:10]}\n")
        
        # Replace any NaNs or infs with 0
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        f.write(f"Values after NaN/inf replacement (first 10): {values[:10]}\n")
        
        # All columns are microbial features, so use braycurtis distance
        dist_mats[col] = compute_distance_matrix(values, metric='braycurtis')
        f.write(f"Distance matrix shape: {dist_mats[col].shape}\n")
        f.write(f"Distance matrix (first 10x10):\n")
        np.savetxt(f, dist_mats[col][:10, :10], fmt='%.4f')
        f.write("\n")
    else:
        # For non-demo nodes, just compute the distance matrix
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
        if i < 5:  # Print first 5 permutations
            f.write(f"Permutation {i+1}: r_perm={r_perm:.4f}\n")
    
    p_val = (count + 1) / (permutations + 1)
    f.write(f"Final p-value: {p_val:.4f}\n")
    return p_val, r_obs

# Perform Mantel test between demo nodes
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

# Continue with the rest of the edge construction
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

# Close the file
f.close()
print("Intermediate results saved to 'intermediate_results/distance_matrices_and_edges.txt'")



#%% Create PyG Data Objects
# graphs = []
# for idx in range(len(feature_matrix)):
#     x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)  
#     # 1) **No** .view(1,2) — store as 1‐D vector of length 2
#     y = torch.tensor(target_data.iloc[idx].values.astype(np.float32), dtype=torch.float)  # shape [2]
#     graphs.append(Data(x=x, edge_index=edge_index, y=y))

# print(f"Created {len(graphs)} graphs with {num_nodes} nodes each.")

# %%Graph Visualization with NetworkX
import networkx as nx
# Graph all the samples
G = nx.Graph()
for i, col in enumerate(node_feature_names):
    G.add_node(col, type='microbe')  # All nodes are now microbial features
    
for src, tgt in zip(edge_index_list[0], edge_index_list[1]):
    G.add_edge(node_feature_names[src], node_feature_names[tgt])
    
# All nodes are now microbe nodes
microbe_nodes = list(G.nodes())

# Use Kamada-Kawai layout for a more balanced view.
pos = nx.kamada_kawai_layout(G)

# Create a larger figure.
plt.figure(figsize=(12,12))
nx.draw_networkx_edges(G, pos, edge_color='#888888', alpha=0.5, width=1)

# Draw microbial nodes (blue circles) with smaller sizes.
nx.draw_networkx_nodes(G, pos, nodelist=microbe_nodes, node_color='skyblue', 
                      node_shape='o', node_size=200, label='Microbial Family')

# Label ALL nodes (not just top 15)
labels_dict = {n: n for n in microbe_nodes}
nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.6))

plt.title("Biogas Interaction Graph - Microbial Families Only")
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.show()

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
    y = torch.tensor(y_vals, dtype=torch.float).unsqueeze(0)  # shape: [2]
    
    # Create graph with edge_index and graph-level target
    graph = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(graph)

print(f"Created {len(graphs)} graph data objects (one per sample).")
print(f"Each graph has {num_nodes} nodes (microbial families) and {edge_index.shape[1]//2} undirected edges.")
print(f"Target dimensions: {graphs[0].y.shape}")

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
        
        # For GCN, use our custom implementation with GCNConv
        if model_name == 'GCN':
            self.use_custom_impl = True
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            # Create layers
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Middle layers
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Activation function
            if act == 'relu':
                self.act_fn = F.relu
            elif act == 'elu':
                self.act_fn = F.elu
            else:
                self.act_fn = F.relu
            
            self.dropout = nn.Dropout(dropout)
        else:
            # For other models, use the high-level wrappers
            self.use_custom_impl = False
            model_map = {
                'GraphSAGE': GraphSAGE,
                'GIN':       GIN,
                'GAT':       GAT,
                'PNA':       PNA,
            }
            Wrapper = model_map[model_name]
            
            self.gnn = Wrapper(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=hidden_channels,
                dropout=dropout,
                act=act,
                jk=jk,
                **kwargs
            )
        
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        if self.use_custom_impl:
            # Custom implementation for GCN
            from torch_geometric.nn import global_mean_pool
            
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Apply GCN layers
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = self.act_fn(x)
                x = self.dropout(x)
            
            # Global pooling
            x = global_mean_pool(x, batch)
        else:
            # High-level wrappers for other models
            x = self.gnn(data.x, data.edge_index, data.batch)
            
            # Apply pooling if needed
            if x.size(0) != data.num_graphs:
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, data.batch)
        
        # Final projection
        return self.head(x)  # [batch_size, out_channels]


#%% Training Function
def train_model(target_idx, target_name,
                model_name='GIN',
                hidden_channels=64,
                num_layers=3,
                dropout=0.3,
                jk=None,
                **kwargs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_trues = [], []

    # Check the structure of a single graph's y
    print(f"Single graph y shape: {graphs[0].y.shape}, dtype: {graphs[0].y.dtype}")
    print(f"First few y values: {graphs[0].y}")

    for fold, (tr, va) in enumerate(kf.split(graphs), 1):
        tr_loader = DataLoader([graphs[i] for i in tr], batch_size=16, shuffle=True)
        va_loader = DataLoader([graphs[i] for i in va], batch_size=16)

        # Get the first batch to inspect
        for debug_batch in tr_loader:
            print(f"Batch y shape: {debug_batch.y.shape}, dtype: {debug_batch.y.dtype}")
            print(f"Batch size: {debug_batch.num_graphs}")
            if len(debug_batch.y.shape) == 1:
                print("Y is 1D tensor - needs to be reshaped")
            else:
                print("Y is already in expected shape")
            # Only checking the first batch
            break

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
        loss_fn   = nn.MSELoss()

        for epoch in range(1, 301):
            model.train()
            total_loss = 0
            for batch in tr_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)  # [B,1]
                
                # Get the target values properly - this is the core fix
                if len(batch.y.shape) == 1:
                    # Shape is flattened to [B*2]
                    batch_size = batch.num_graphs
                    if batch.y.size(0) == batch_size * 2:
                        # Extract every other element starting from target_idx
                        target = batch.y[target_idx::2].unsqueeze(-1)  # [B,1]
                    else:
                        # Alternative approach if the above doesn't work
                        batch_y_reshaped = batch.y.view(batch_size, -1)
                        target = batch_y_reshaped[:, target_idx].unsqueeze(-1)
                else:
                    # Already in expected [B,2] format
                    target = batch.y[:, target_idx].unsqueeze(-1)

                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            
            if epoch % 50 == 0:
                print(f"{target_name} [{model_name}] Fold {fold} Epoch {epoch} Train MSE: {total_loss/len(tr_loader.dataset):.4f}")

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                p = model(batch)
                
                # Same logic as in training loop
                if len(batch.y.shape) == 1:
                    batch_size = batch.num_graphs
                    if batch.y.size(0) == batch_size * 2:
                        t = batch.y[target_idx::2].unsqueeze(-1)
                    else:
                        batch_y_reshaped = batch.y.view(batch_size, -1)
                        t = batch_y_reshaped[:, target_idx].unsqueeze(-1)
                else:
                    t = batch.y[:, target_idx].unsqueeze(-1)
                
                preds.append(p.cpu().numpy())
                trues.append(t.cpu().numpy())
        
        all_preds.append(np.vstack(preds))
        all_trues.append(np.vstack(trues))
        print(f"{target_name} [{model_name}] Fold {fold} done.")

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    mse = mean_squared_error(all_trues, all_preds)
    r2  = r2_score(all_trues, all_preds)
    print(f"\n{target_name} [{model_name}] CV MSE: {mse:.4f}, R^2: {r2:.3f}")

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


#%% Train Separate Models
print("=== Training ACE-km ===")

# For PNA model, we need to calculate the degree histogram
if 'PNA' in ['PNA', 'pna']:
    # Compute in-degree histogram over training data
    from torch_geometric.utils import degree
    deg = torch.zeros(100, dtype=torch.long)
    for data in graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

ace_preds, ace_trues, ace_mse, ace_r2 = train_model(
    target_idx=0,
    target_name="ACE-km",
    model_name='GIN',
    hidden_channels=64,  # Increased from 64
    num_layers=4,         # Increased from 3
    dropout=0.2,          # Decreased from 0.6
    # jk='cat',
    # heads=4   # extra arg for GAT
    # Required PNA parameters
    # aggregators=['mean', 'min', 'max', 'std'],
    # scalers=['identity', 'amplification', 'attenuation'],
    # deg=deg
)

print("\n=== Training H2-km ===")
h2_preds, h2_trues, h2_mse, h2_r2 = train_model(
    target_idx=1,
    target_name="H2-km",
    model_name='GIN',
    hidden_channels=64,  # Increased from 64
    num_layers=4,         # Increased from 3
    dropout=0.2,          # Decreased from 0.6
    # jk='cat',
    # heads=4   # extra arg for GAT
    # Required PNA parameters
    # aggregators=['mean', 'min', 'max', 'std'],
    # scalers=['identity', 'amplification', 'attenuation'],
    # deg=deg
) 
# %%

