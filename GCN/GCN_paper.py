# import os
# import time
# import math
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import GCNConv
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error
# import networkx as nx
# import matplotlib.pyplot as plt

# ###############################################################################
# # 1) DATA PREPROCESSING & FILTERING (Only Taxa and Target Columns)
# ###############################################################################

# def preprocess_data(csv_path, abundance_threshold=0.01, prevalence_threshold=0.05, taxon_map=None):
#     """
#     Loads and preprocesses the dataset from `csv_path` following Kim et al. (2025):
#       1. Filter out low-abundance OTUs (keep OTUs with >1% relative abundance in >=5% of samples).
#       2. (Optionally) Merge OTUs at the family level using `taxon_map`.
#       3. Apply double-square-root transform to microbial relative abundances.
#       4. Log-transform and standardize target variables.
#       5. Standardize all features.
      
#     Returns:
#       df_filtered: DataFrame with columns = [taxa columns] + [ACE-km, H2-km]
#       taxa_cols: List of filtered OTU (or merged family) names.
#       target_cols: List of target names (["ACE-km", "H2-km"])
#     """
#     print("Loading data from:", csv_path)
#     df = pd.read_csv(csv_path)
    
#     # Define target columns
#     target_cols = ["ACE-km", "H2-km"]
    
#     # Assume all columns except target and potential ID/time columns are OTU columns
#     otu_cols = [col for col in df.columns if col not in target_cols and col not in ["Sample", "SampleID", "Time", "Day"]]
    
#     # Remove rows with contamination (e.g., containing "x" in any column)
#     for col in otu_cols + target_cols:
#         df = df[df[col].astype(str).str.contains('x', case=False) == False]
    
#     # Convert OTU columns to numeric; drop rows that cannot be converted
#     for col in otu_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.dropna(subset=otu_cols, inplace=True)
#     df[otu_cols] = df[otu_cols].fillna(0)
    
#     # Also ensure target columns are numeric; drop rows where target conversion fails.
#     for col in target_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.dropna(subset=target_cols, inplace=True)
    
#     # Convert OTU counts to relative abundances (row-wise)
#     otu_data = df[otu_cols].copy()
#     otu_sums = otu_data.sum(axis=1)
#     otu_sums[otu_sums == 0] = 1e-9
#     otu_data = otu_data.div(otu_sums, axis=0)
    
#     # Filter OTUs by prevalence threshold
#     keep_otus = []
#     for col in otu_cols:
#         frac_present = (otu_data[col] > abundance_threshold).mean()
#         if frac_present >= prevalence_threshold:
#             keep_otus.append(col)
#     print(f"Filtered OTUs: {len(otu_cols)} -> {len(keep_otus)} based on thresholds.")
    
#     # Keep only filtered OTUs and target columns
#     df_filtered = df[keep_otus + target_cols].copy()
    
#     # Optional: Merge OTUs by taxonomy if taxon_map provided
#     if taxon_map is not None:
#         filtered_map = {otu: taxon_map[otu] for otu in keep_otus if otu in taxon_map}
#         if filtered_map:
#             df_taxon = df_filtered.groupby(filtered_map, axis=1).sum()
#             for tc in target_cols:
#                 if tc in df_filtered.columns:
#                     df_taxon[tc] = df_filtered[tc]
#             df_filtered = df_taxon
#             keep_otus = list(set(filtered_map.values()))
    
#     # Transformation: Apply double-square-root to taxa nodes
#     for col in keep_otus:
#         df_filtered[col] = df_filtered[col].astype(float)
#         df_filtered[col] = np.sqrt(np.sqrt(df_filtered[col]))
    
#     # Log-transform target columns (if numeric)
#     for col in target_cols:
#         if np.issubdtype(df_filtered[col].dtype, np.number):
#             df_filtered[col] = np.log1p(df_filtered[col].clip(lower=0))
    
#     # Standardize all columns (z-score)
#     all_feat_cols = keep_otus + target_cols
#     for col in all_feat_cols:
#         mean_val = df_filtered[col].mean()
#         std_val = df_filtered[col].std(ddof=0)
#         if std_val != 0:
#             df_filtered[col] = (df_filtered[col] - mean_val) / std_val
    
#     print("Preprocessing complete. Data shape:", df_filtered.shape)
#     return df_filtered, keep_otus, target_cols

# ###############################################################################
# # 2) NODE & EDGE DEFINITION USING MANTEL TEST
# ###############################################################################

# def mantel_test(dist_matrix1, dist_matrix2, permutations=999):
#     """
#     Compute the Mantel test correlation between two condensed distance vectors.
#     Returns (r_obs, p_value).
#     """
#     r_obs, _ = pearsonr(dist_matrix1, dist_matrix2)
#     perm_count = 0
#     size = dist_matrix1.shape[0]
#     n = int(np.ceil((1 + np.sqrt(1 + 8 * size)) / 2))
#     dist_mat2_full = squareform(dist_matrix2)
#     for _ in range(permutations):
#         perm_idx = np.random.permutation(n)
#         dist_mat2_perm = dist_mat2_full[perm_idx][:, perm_idx]
#         dist_mat2_perm_cond = squareform(dist_mat2_perm)
#         r_perm, _ = pearsonr(dist_matrix1, dist_mat2_perm_cond)
#         if r_perm >= r_obs:
#             perm_count += 1
#     p_value = (perm_count + 1) / (permutations + 1)
#     return r_obs, p_value

# def build_graph_edges(df, taxa_cols, target_cols, alpha=0.05):
#     """
#     Build a static interaction network.
#       - Node set = taxa nodes + target nodes.
#       - For taxa nodes, compute Bray–Curtis distances.
#       - For target nodes, use Euclidean distances.
#       - For each pair of nodes, run Mantel tests (999 permutations) and create an edge if p < alpha.
      
#     Returns:
#       feature_list: list of node names (taxa + targets)
#       edge_index: torch.LongTensor of shape [2, E]
#       edge_weight: torch.FloatTensor of shape [E]
#     """
#     features = taxa_cols + target_cols
#     num_features = len(features)
#     dist_vectors = {}
#     for feat in features:
#         vals = pd.to_numeric(df[feat], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
#         if feat in taxa_cols:
#             dvec = pdist(vals, metric="braycurtis")
#         else:
#             dvec = pdist(vals, metric="euclidean")
#         dist_vectors[feat] = dvec

#     edges = []
#     edge_weights = []
#     for i in range(num_features):
#         for j in range(i + 1, num_features):
#             r_val, p_val = mantel_test(dist_vectors[features[i]], dist_vectors[features[j]], permutations=999)
#             if p_val < alpha:
#                 edges.append((i, j))
#                 edges.append((j, i))
#                 edge_weights.append(r_val)
#                 edge_weights.append(r_val)
#     if not edges:
#         raise ValueError("No significant edges found. Adjust thresholds or check data.")
#     edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
#     edge_weight = torch.tensor(edge_weights, dtype=torch.float)
#     print(f"Graph topology: {num_features} nodes, {edge_index.shape[1]} edges.")
#     return features, edge_index, edge_weight

# def visualize_graph(feature_list, edge_index, save_path="graph_topology.png"):
#     """
#     Visualize the constructed graph using NetworkX and save the figure.
#     """
#     G = nx.Graph()
#     G.add_nodes_from(range(len(feature_list)))
#     edge_list = edge_index.t().tolist()
#     G.add_edges_from(edge_list)
#     labels = {i: feature_list[i] for i in range(len(feature_list))}
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(G, seed=42)
#     nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500)
#     nx.draw_networkx_edges(G, pos, alpha=0.7)
#     nx.draw_networkx_labels(G, pos, labels, font_size=9)
#     plt.title("Graph Topology (Taxa Nodes + Target Nodes)")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Graph visualization saved to {save_path}")

# ###############################################################################
# # 3) BUILDING INDIVIDUAL GRAPHS (ONE PER SAMPLE)
# ###############################################################################

# def create_graph_dataset(df, feature_list, edge_index, edge_weight, target_cols):
#     """
#     Create one PyG Data object per sample.
#       - x: Node features [num_nodes, 1] from the sample.
#              For target nodes, set input value to 0 (to be predicted).
#       - y: The true target values (for ACE-km and H2-km).
#     """
#     dataset = []
#     target_indices = [feature_list.index(tc) for tc in target_cols]
#     for _, row in df.iterrows():
#         x_vals = row[feature_list].to_numpy(dtype=float)
#         # Mask target nodes in input by setting their values to 0.
#         for ti in target_indices:
#             x_vals[ti] = 0.0
#         x_tensor = torch.tensor(x_vals, dtype=torch.float).view(-1, 1)
#         y_vals = row[target_cols].to_numpy(dtype=float)
#         y_tensor = torch.tensor(y_vals, dtype=torch.float)
#         data = Data(x=x_tensor, edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), y=y_tensor)
#         dataset.append(data)
#     print(f"Created {len(dataset)} graph samples.")
#     return dataset

# ###############################################################################
# # 4) GCN MODEL DEFINITION
# ###############################################################################

# class GCNPredictor(nn.Module):
#     """
#     A simple GCN-based predictor with two GCNConv layers.
#     It extracts embeddings for target nodes and predicts their values.
#     """
#     def __init__(self, num_nodes, input_dim, hidden_dim, target_indices):
#         super(GCNPredictor, self).__init__()
#         self.target_indices = target_indices
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.fc_ace = nn.Linear(hidden_dim, 1)
#         self.fc_h2 = nn.Linear(hidden_dim, 1)
    
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.relu(self.conv2(x, edge_index, edge_weight))
#         ace_idx, h2_idx = self.target_indices
#         ace_embed = x[ace_idx]
#         h2_embed = x[h2_idx]
#         ace_pred = self.fc_ace(ace_embed)
#         h2_pred = self.fc_h2(h2_embed)
#         out = torch.cat([ace_pred, h2_pred], dim=0)
#         return out

# ###############################################################################
# # 5) TRAINING FUNCTION WITH BATCHING & GPU SUPPORT
# ###############################################################################

# def train_gcn_model(model, train_dataset, test_dataset, epochs=100, lr=0.005, batch_size=1):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     print(f"Training on device: {device}")
#     for epoch in range(1, epochs + 1):
#         model.train()
#         epoch_loss = 0.0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             pred = model(batch)
#             loss = loss_fn(pred, batch.y)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * batch.num_graphs
#         avg_loss = epoch_loss / len(train_dataset)
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch:3d} - Training MSE: {avg_loss:.4f}")
    
#     model.eval()
#     all_preds, all_trues = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#             out = model(batch)
#             all_preds.append(out.cpu().numpy())
#             all_trues.append(batch.y.cpu().numpy())
#     preds = np.vstack(all_preds)
#     trues = np.vstack(all_trues)
#     mse_value = mean_squared_error(trues, preds)
#     print(f"Test MSE: {mse_value:.4f}")
#     return model, mse_value

# ###############################################################################
# # 6) MAIN EXECUTION
# ###############################################################################

# def main():
#     start_time = time.time()
#     csv_path = "Data/New_data.csv"  # Update the path if needed
#     print("Starting data preprocessing...")
#     df_filtered, taxa_cols, target_cols = preprocess_data(
#         csv_path,
#         abundance_threshold=0.01,
#         prevalence_threshold=0.05,
#         taxon_map=None  # Provide mapping dict if available
#     )
#     print("Building graph edges...")
#     feature_list, edge_index, edge_weight = build_graph_edges(
#         df_filtered, taxa_cols, target_cols, alpha=0.05
#     )
#     visualize_graph(feature_list, edge_index, save_path="graph_topology.png")
    
#     print("Creating graph dataset...")
#     dataset = create_graph_dataset(
#         df_filtered,
#         feature_list,
#         edge_index,
#         edge_weight,
#         target_cols=target_cols
#     )
#     np.random.shuffle(dataset)
#     split_idx = int(0.8 * len(dataset))
#     train_data = dataset[:split_idx]
#     test_data = dataset[split_idx:]
#     target_indices = [feature_list.index(tc) for tc in target_cols]
#     print("Initializing GCN model...")
#     model = GCNPredictor(
#         num_nodes=len(feature_list),
#         input_dim=1,
#         hidden_dim=32,
#         target_indices=target_indices
#     )
#     print("Training GCN model...")
#     trained_model, test_mse = train_gcn_model(
#         model,
#         train_data,
#         test_data,
#         epochs=100,
#         lr=0.005,
#         batch_size=1
#     )
#     elapsed = time.time() - start_time
#     print(f"Training complete in {elapsed/60:.2f} minutes. Final Test MSE: {test_mse:.4f}")

# if __name__ == "__main__":
#     main()




## Complete code
# import os
# import time
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import GCNConv
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error, r2_score
# import networkx as nx
# import matplotlib.pyplot as plt

# ###############################################################################
# # 1) DATA PREPROCESSING (Using OTU columns that start with "d__")
# ###############################################################################

# def preprocess_data(csv_path, abundance_threshold=0.001, prevalence_threshold=0.01, otu_prefix="d__"):
#     """
#     Loads and preprocesses the dataset from `csv_path` following Kim et al. (2025):
#       - Considers columns starting with `otu_prefix` as OTU columns.
#       - Keeps also the target columns: "ACE-km" and "H2-km".
#       - Filters out low-abundance OTUs (keep OTUs with >abundance_threshold relative abundance in >=prevalence_threshold of samples).
#       - Applies a double-square-root transform to OTU abundances.
#       - Leaves the target columns unchanged.
#       - Standardizes OTU columns.
      
#     Returns:
#       df_filtered: DataFrame with columns = [filtered OTUs] + [ACE-km, H2-km]
#       taxa_cols: List of filtered OTU column names.
#       target_cols: List of target names (["ACE-km", "H2-km"])
#     """
#     print("Loading data from:", csv_path)
#     df = pd.read_csv(csv_path)
    
#     target_cols = ["ACE-km", "H2-km"]
#     all_cols = df.columns.tolist()
#     otu_cols = [c for c in all_cols if c.startswith(otu_prefix)]
#     keep_cols = otu_cols + target_cols
    
#     # Remove rows with non-numeric contamination (e.g., containing "x")
#     for col in keep_cols:
#         df = df[df[col].astype(str).str.contains('x', case=False) == False]
    
#     # Keep only the specified columns
#     df = df[[c for c in keep_cols if c in df.columns]].copy()
    
#     # Convert OTU columns to numeric; drop rows that can't be converted
#     for col in otu_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.dropna(subset=otu_cols, inplace=True)
#     df[otu_cols] = df[otu_cols].fillna(0)
    
#     # Convert target columns to numeric; drop rows that fail conversion
#     for col in target_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.dropna(subset=target_cols, inplace=True)
    
#     # Convert OTU counts to relative abundances (row-wise)
#     otu_data = df[otu_cols].copy()
#     sums = otu_data.sum(axis=1)
#     sums[sums == 0] = 1e-9
#     otu_data = otu_data.div(sums, axis=0)
    
#     # Filter OTUs by prevalence threshold
#     keep_otus = []
#     for col in otu_cols:
#         frac_present = (otu_data[col] > abundance_threshold).mean()
#         if frac_present >= prevalence_threshold:
#             keep_otus.append(col)
#     print(f"Filtered OTUs: {len(otu_cols)} -> {len(keep_otus)} based on thresholds.")
    
#     # Keep only filtered OTUs and target columns
#     final_cols = keep_otus + target_cols
#     df_filtered = df[final_cols].copy()
    
#     if len(keep_otus) == 0:
#         print("WARNING: No OTUs remain after filtering! Only targets remain.")
    
#     # Apply double-square-root transform to OTU columns
#     for col in keep_otus:
#         df_filtered[col] = np.sqrt(np.sqrt(df_filtered[col].astype(float)))
#         # Standardize OTU columns (z-score)
#         mean_val = df_filtered[col].mean()
#         std_val = df_filtered[col].std(ddof=0)
#         if std_val != 0:
#             df_filtered[col] = (df_filtered[col] - mean_val) / std_val
    
#     print("Preprocessing complete. Data shape:", df_filtered.shape)
#     return df_filtered, keep_otus, target_cols

# ###############################################################################
# # 2) NODE & EDGE DEFINITION USING MANTEL TEST
# ###############################################################################

# def mantel_test(dist_matrix1, dist_matrix2, permutations=999):
#     r_obs, _ = pearsonr(dist_matrix1, dist_matrix2)
#     perm_count = 0
#     size = dist_matrix1.shape[0]
#     n = int(np.ceil((1 + np.sqrt(1 + 8 * size)) / 2))
#     dist_mat2_full = squareform(dist_matrix2)
#     for _ in range(permutations):
#         perm_idx = np.random.permutation(n)
#         dist_mat2_perm = dist_mat2_full[perm_idx][:, perm_idx]
#         dist_mat2_perm_cond = squareform(dist_mat2_perm)
#         r_perm, _ = pearsonr(dist_matrix1, dist_mat2_perm_cond)
#         if r_perm >= r_obs:
#             perm_count += 1
#     p_value = (perm_count + 1) / (permutations + 1)
#     return r_obs, p_value

# def build_graph_edges(df, taxa_cols, target_cols, alpha=0.05):
#     """
#     Build a static graph with nodes = taxa (OTUs) + targets.
#       - For taxa nodes, uses Bray–Curtis distances.
#       - For target nodes, uses Euclidean distances.
#       - Runs Mantel tests (999 perms); adds edge if p < alpha.
      
#     Returns:
#       feature_list: list of node names (taxa followed by targets).
#       edge_index: torch.LongTensor of shape [2, E].
#       edge_weight: torch.FloatTensor of shape [E].
#     """
#     features = taxa_cols + target_cols
#     dist_vectors = {}
#     for feat in features:
#         vals = df[feat].to_numpy().reshape(-1, 1)
#         if feat in taxa_cols:
#             dvec = pdist(vals, metric="braycurtis")
#         else:
#             dvec = pdist(vals, metric="euclidean")
#         dist_vectors[feat] = dvec
    
#     edges = []
#     weights = []
#     for i in range(len(features)):
#         for j in range(i+1, len(features)):
#             r_val, p_val = mantel_test(dist_vectors[features[i]], dist_vectors[features[j]], permutations=999)
#             if p_val < alpha:
#                 edges.append((i, j))
#                 edges.append((j, i))
#                 weights.append(r_val)
#                 weights.append(r_val)
    
#     if not edges:
#         raise ValueError("No significant edges found. Adjust thresholds or check data.")
    
#     edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
#     edge_weight = torch.tensor(weights, dtype=torch.float)
#     print(f"Constructed graph with {len(features)} nodes and {edge_index.shape[1]} edges.")
#     return features, edge_index, edge_weight

# def visualize_graph_static(feature_list, edge_index, save_path="graph_topology.png"):
#     G = nx.Graph()
#     G.add_nodes_from(range(len(feature_list)))
#     edge_list = edge_index.t().tolist() if edge_index.shape[1] > 0 else []
#     G.add_edges_from(edge_list)
#     labels = {i: feature_list[i] for i in range(len(feature_list))}
    
#     plt.figure(figsize=(8,6))
#     pos = nx.spring_layout(G, seed=42)
#     nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=600)
#     nx.draw_networkx_edges(G, pos, alpha=0.7)
#     nx.draw_networkx_labels(G, pos, labels, font_size=8)
#     plt.title("Static Graph Topology (Taxa + Targets)")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved static graph topology to {save_path}")

# def visualize_sample_graph(data, feature_list, sample_idx, outdir="graphs"):
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#     G = nx.Graph()
#     n_nodes = data.x.shape[0]
#     G.add_nodes_from(range(n_nodes))
#     edge_list = data.edge_index.t().tolist() if data.edge_index.shape[1] > 0 else []
#     G.add_edges_from(edge_list)
#     node_values = data.x.view(-1).cpu().numpy()
#     labels = {i: feature_list[i] for i in range(n_nodes)}
    
#     plt.figure(figsize=(8,6))
#     pos = nx.spring_layout(G, seed=42)
#     nodes = nx.draw_networkx_nodes(G, pos, node_color=node_values, cmap=plt.cm.viridis, node_size=600)
#     nx.draw_networkx_edges(G, pos, alpha=0.7)
#     nx.draw_networkx_labels(G, pos, labels, font_size=8)
#     plt.colorbar(nodes, label="Node Feature Value")
#     plt.title(f"Sample Graph #{sample_idx}")
#     plt.axis("off")
#     plt.tight_layout()
#     outpath = os.path.join(outdir, f"sample_graph_{sample_idx}.png")
#     plt.savefig(outpath)
#     plt.close()
#     print(f"Saved sample graph {sample_idx} to {outpath}")

# ###############################################################################
# # 3) CREATE ONE GRAPH PER SAMPLE
# ###############################################################################

# def create_dataset(df, feature_list, edge_index, edge_weight, target_cols, save_individual=False):
#     dataset = []
#     target_indices = [feature_list.index(tc) for tc in target_cols]
    
#     for idx, row in df.iterrows():
#         x_vals = row[feature_list].to_numpy(dtype=float)
#         # Mask target nodes (set to 0)
#         for ti in target_indices:
#             x_vals[ti] = 0.0
#         x_tensor = torch.tensor(x_vals, dtype=torch.float).view(-1, 1)
#         y_vals = row[target_cols].to_numpy(dtype=float)
#         y_tensor = torch.tensor(y_vals, dtype=torch.float)
#         data = Data(x=x_tensor, edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), y=y_tensor)
#         dataset.append(data)
#         if save_individual:
#             visualize_sample_graph(data, feature_list, sample_idx=idx, outdir="graphs")
    
#     print(f"Created {len(dataset)} graphs (one per sample).")
#     return dataset

# ###############################################################################
# # 4) SOPHISTICATED GCN MODEL
# ###############################################################################

# class GCNPredictor(nn.Module):
#     def __init__(self, num_nodes, input_dim, hidden_dim, target_indices, dropout=0.3):
#         super().__init__()
#         self.target_indices = target_indices
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
#         self.fc_ace = nn.Linear(hidden_dim, 1)
#         self.fc_h2 = nn.Linear(hidden_dim, 1)
    
#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.relu(self.conv2(x, edge_index, edge_weight))
#         x = F.relu(self.conv3(x, edge_index, edge_weight))
#         x = self.dropout(x)
#         x = F.relu(self.fc_hidden(x))
        
#         ace_idx, h2_idx = self.target_indices
#         ace_embed = x[ace_idx]
#         h2_embed = x[h2_idx]
#         ace_pred = self.fc_ace(ace_embed)
#         h2_pred = self.fc_h2(h2_embed)
#         out = torch.cat([ace_pred, h2_pred], dim=0)
#         return out

# ###############################################################################
# # 5) TRAINING FUNCTION (WITH MSE + R^2)
# ###############################################################################

# def train_model(model, train_data, test_data, epochs=100, lr=0.005):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()
    
#     train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
#     print(f"Training on device: {device}")
#     for epoch in range(1, epochs+1):
#         model.train()
#         total_loss = 0.0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             pred = model(batch)
#             loss = loss_fn(pred, batch.y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss / len(train_data)
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch:3d} | Train MSE: {avg_loss:.4f}")
    
#     model.eval()
#     preds, trues = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#             out = model(batch)
#             preds.append(out.cpu().numpy())
#             trues.append(batch.y.cpu().numpy())
#     preds = np.vstack(preds)
#     trues = np.vstack(trues)
#     mse_val = mean_squared_error(trues, preds)
#     r2_val = r2_score(trues, preds)
#     print(f"Test MSE: {mse_val:.4f} | Test R^2: {r2_val:.4f}")
#     return model, mse_val, r2_val

# ###############################################################################
# # 6) MAIN EXECUTION
# ###############################################################################

# def main():
#     csv_path = "Data/New_data.csv"  # Update path if needed
#     start_time = time.time()
    
#     print("Starting data preprocessing...")
#     df_filtered, taxa_cols, target_cols = preprocess_data(
#         csv_path,
#         abundance_threshold=0.001,  # Adjusted threshold to retain more OTUs if needed
#         prevalence_threshold=0.01,
#         otu_prefix="d__"  # Use columns starting with "d__"
#     )
#     print("Building graph edges...")
#     feat_list, edge_index, edge_weight = build_graph_edges(
#         df_filtered, taxa_cols, target_cols, alpha=0.05
#     )
#     visualize_graph_static(feat_list, edge_index, save_path="graph_topology.png")
    
#     print("Creating graph dataset (saving individual sample visuals)...")
#     dataset = create_dataset(
#         df_filtered,
#         feat_list,
#         edge_index,
#         edge_weight,
#         target_cols,
#         save_individual=True
#     )
#     np.random.shuffle(dataset)
#     split_idx = int(0.8 * len(dataset))
#     train_data = dataset[:split_idx]
#     test_data = dataset[split_idx:]
    
#     target_indices = [feat_list.index(tc) for tc in target_cols]
#     print("Initializing sophisticated GCN model...")
#     model = GCNPredictor(
#         num_nodes=len(feat_list),
#         input_dim=1,
#         hidden_dim=64,
#         target_indices=target_indices,
#         dropout=0.3
#     )
#     print("Training GCN model...")
#     trained_model, mse_val, r2_val = train_model(
#         model,
#         train_data,
#         test_data,
#         epochs=120,
#         lr=0.005
#     )
#     elapsed = time.time() - start_time
#     print(f"\nTraining complete in {elapsed/60:.2f} minutes. Final Test MSE: {mse_val:.4f}, R^2: {r2_val:.4f}")

# if __name__ == "__main__":
#     main()

## Partial run to check what is wrong with the code
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
import matplotlib.pyplot as plt

###############################################################################
# 1) DATA PREPROCESSING (Using OTU columns that start with "d__")
###############################################################################

def preprocess_data(csv_path, abundance_threshold=0.001, prevalence_threshold=0.01, otu_prefix="d__"):
    """
    Loads and preprocesses the dataset from `csv_path`:
      - OTU columns: those starting with `otu_prefix`.
      - Keeps target columns: "ACE-km" and "H2-km".
      - Filters out low-abundance OTUs (keep OTUs with >abundance_threshold in >=prevalence_threshold of samples).
      - Applies double-square-root transform to OTU abundances.
      - Standardizes OTU columns.
      
    Returns:
      df_filtered: DataFrame with columns = [filtered OTUs] + [ACE-km, H2-km]
      taxa_cols: List of filtered OTU column names.
      target_cols: List of target names (["ACE-km", "H2-km"])
    """
    print("Loading data from:", csv_path)
    df = pd.read_csv(csv_path)
    
    target_cols = ["ACE-km", "H2-km"]
    all_cols = df.columns.tolist()
    otu_cols = [c for c in all_cols if c.startswith(otu_prefix)]
    keep_cols = otu_cols + target_cols
    
    # Remove rows with non-numeric contamination
    for col in keep_cols:
        df = df[df[col].astype(str).str.contains('x', case=False) == False]
    
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    
    # Convert OTU columns to numeric
    for col in otu_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=otu_cols, inplace=True)
    df[otu_cols] = df[otu_cols].fillna(0)
    
    # Convert target columns to numeric
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=target_cols, inplace=True)
    
    # Convert OTU counts to relative abundances
    otu_data = df[otu_cols].copy()
    sums = otu_data.sum(axis=1)
    sums[sums == 0] = 1e-9
    otu_data = otu_data.div(sums, axis=0)
    
    # Filter OTUs by prevalence
    keep_otus = []
    for col in otu_cols:
        frac_present = (otu_data[col] > abundance_threshold).mean()
        if frac_present >= prevalence_threshold:
            keep_otus.append(col)
    print(f"Filtered OTUs: {len(otu_cols)} -> {len(keep_otus)} based on thresholds.")
    
    final_cols = keep_otus + target_cols
    df_filtered = df[final_cols].copy()
    
    if len(keep_otus) == 0:
        print("WARNING: No OTUs remain after filtering! Only target columns remain.")
    
    # Apply double-square-root transform to OTU columns
    for col in keep_otus:
        df_filtered[col] = np.sqrt(np.sqrt(df_filtered[col].astype(float)))
        mean_val = df_filtered[col].mean()
        std_val = df_filtered[col].std(ddof=0)
        if std_val != 0:
            df_filtered[col] = (df_filtered[col] - mean_val) / std_val
    
    print("Preprocessing complete. Data shape:", df_filtered.shape)
    return df_filtered, keep_otus, target_cols

###############################################################################
# 2) NODE & EDGE DEFINITION USING MANTEL TEST
###############################################################################

def mantel_test(dist_matrix1, dist_matrix2, permutations=999):
    r_obs, _ = pearsonr(dist_matrix1, dist_matrix2)
    perm_count = 0
    size = dist_matrix1.shape[0]
    n = int(np.ceil((1 + np.sqrt(1 + 8 * size)) / 2))
    dist_mat2_full = squareform(dist_matrix2)
    for _ in range(permutations):
        perm_idx = np.random.permutation(n)
        dist_mat2_perm = dist_mat2_full[perm_idx][:, perm_idx]
        dist_mat2_perm_cond = squareform(dist_mat2_perm)
        r_perm, _ = pearsonr(dist_matrix1, dist_mat2_perm_cond)
        if r_perm >= r_obs:
            perm_count += 1
    p_value = (perm_count + 1) / (permutations + 1)
    return r_obs, p_value

def build_graph_edges(df, taxa_cols, target_cols, alpha=0.05):
    """
    Build a static graph with nodes = taxa (OTUs) + targets.
      - For taxa nodes, compute Bray–Curtis distances.
      - For target nodes, compute Euclidean distances.
      - Use Mantel tests to add an edge if p < alpha.
      
    Returns:
      feature_list: list of node names.
      edge_index: torch.LongTensor [2, E].
      edge_weight: torch.FloatTensor [E].
    """
    features = taxa_cols + target_cols
    dist_vectors = {}
    for feat in features:
        vals = df[feat].to_numpy().reshape(-1, 1)
        if feat in taxa_cols:
            dvec = pdist(vals, metric="braycurtis")
        else:
            dvec = pdist(vals, metric="euclidean")
        dist_vectors[feat] = dvec
    
    edges = []
    weights = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            r_val, p_val = mantel_test(dist_vectors[features[i]], dist_vectors[features[j]], permutations=999)
            if p_val < alpha:
                edges.append((i, j))
                edges.append((j, i))
                weights.append(r_val)
                weights.append(r_val)
    
    if not edges:
        raise ValueError("No significant edges found. Adjust thresholds or check data.")
    
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    print(f"Constructed graph with {len(features)} nodes and {edge_index.shape[1]} edges.")
    return features, edge_index, edge_weight

def visualize_graph_static(feature_list, edge_index, save_path="graph_topology.png"):
    G = nx.Graph()
    G.add_nodes_from(range(len(feature_list)))
    if edge_index.shape[1] > 0:
        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)
    labels = {i: feature_list[i] for i in range(len(feature_list))}
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=600)
    nx.draw_networkx_edges(G, pos, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("Static Graph Topology (Taxa + Targets)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved static graph topology to {save_path}")

###############################################################################
# 3) CREATE ONE GRAPH PER SAMPLE (DEBUG MODE: Use only one sample)
###############################################################################

def visualize_sample_graph(data, feature_list, sample_idx, outdir="graphs"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    G = nx.Graph()
    n_nodes = data.x.shape[0]
    G.add_nodes_from(range(n_nodes))
    if data.edge_index.shape[1] > 0:
        edge_list = data.edge_index.t().tolist()
        G.add_edges_from(edge_list)
    node_values = data.x.view(-1).cpu().numpy()
    labels = {i: feature_list[i] for i in range(n_nodes)}
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_values, cmap=plt.cm.viridis, node_size=600)
    nx.draw_networkx_edges(G, pos, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.colorbar(nodes, label="Node Feature Value")
    plt.title(f"Sample Graph #{sample_idx}")
    plt.axis("off")
    plt.tight_layout()
    outpath = os.path.join(outdir, f"sample_graph_{sample_idx}.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved sample graph {sample_idx} to {outpath}")

def create_dataset(df, feature_list, edge_index, edge_weight, target_cols, save_individual=False):
    dataset = []
    target_indices = [feature_list.index(tc) for tc in target_cols]
    for idx, row in df.iterrows():
        x_vals = row[feature_list].to_numpy(dtype=float)
        for ti in target_indices:
            x_vals[ti] = 0.0  # mask target nodes
        x_tensor = torch.tensor(x_vals, dtype=torch.float).view(-1, 1)
        y_vals = row[target_cols].to_numpy(dtype=float)
        y_tensor = torch.tensor(y_vals, dtype=torch.float)
        data = Data(x=x_tensor, edge_index=edge_index.clone(), edge_weight=edge_weight.clone(), y=y_tensor)
        dataset.append(data)
        if save_individual:
            visualize_sample_graph(data, feature_list, sample_idx=idx, outdir="graphs")
    print(f"Created {len(dataset)} graphs (one per sample).")
    return dataset

###############################################################################
# 4) SOPHISTICATED GCN MODEL
###############################################################################

class GCNPredictor(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, target_indices, dropout=0.3):
        super().__init__()
        self.target_indices = target_indices
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_ace = nn.Linear(hidden_dim, 1)
        self.fc_h2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.fc_hidden(x))
        ace_idx, h2_idx = self.target_indices
        ace_embed = x[ace_idx]
        h2_embed = x[h2_idx]
        ace_pred = self.fc_ace(ace_embed)
        h2_pred = self.fc_h2(h2_embed)
        out = torch.cat([ace_pred, h2_pred], dim=0)
        return out

###############################################################################
# 5) TRAINING FUNCTION (With MSE and R^2)
###############################################################################

def train_model(model, train_data, test_data, epochs=100, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    print(f"Training on device: {device}")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train MSE: {avg_loss:.4f}")
    
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mse_val = mean_squared_error(trues, preds)
    r2_val = r2_score(trues, preds)
    print(f"Test MSE: {mse_val:.4f} | Test R^2: {r2_val:.4f}")
    return model, mse_val, r2_val

###############################################################################
# 6) MAIN EXECUTION (DEBUG MODE: Use 1 Graph Only)
###############################################################################

def main():
    csv_path = "Data/New_data.csv"  # Update path if needed
    start_time = time.time()
    
    print("Starting data preprocessing...")
    df_filtered, taxa_cols, target_cols = preprocess_data(
        csv_path,
        abundance_threshold=0.001,
        prevalence_threshold=0.01,
        otu_prefix="d__"  # Use columns starting with "d__"
    )
    
    print("Building graph edges...")
    feat_list, edge_index, edge_weight = build_graph_edges(
        df_filtered, taxa_cols, target_cols, alpha=0.05
    )
    visualize_graph_static(feat_list, edge_index, save_path="graph_topology.png")
    
    print("Creating dataset (DEBUG: Using only first sample)...")
    full_dataset = create_dataset(
        df_filtered,
        feat_list,
        edge_index,
        edge_weight,
        target_cols,
        save_individual=True
    )
    # DEBUG: Use only one graph for both training and testing
    if len(full_dataset) > 0:
        dataset = full_dataset[:1]
    else:
        raise ValueError("No graphs were created.")
    
    train_data = dataset
    test_data = dataset
    
    target_indices = [feat_list.index(tc) for tc in target_cols]
    print("Initializing sophisticated GCN model...")
    model = GCNPredictor(
        num_nodes=len(feat_list),
        input_dim=1,
        hidden_dim=64,
        target_indices=target_indices,
        dropout=0.3
    )
    
    print("Training model on one graph...")
    trained_model, mse_val, r2_val = train_model(
        model,
        train_data,
        test_data,
        epochs=1,   # Fewer epochs for debugging
        lr=0.005
    )
    elapsed = time.time() - start_time
    print(f"\nDebug run complete in {elapsed/60:.2f} minutes. Final Test MSE: {mse_val:.4f}, R^2: {r2_val:.4f}")

if __name__ == "__main__":
    main()
