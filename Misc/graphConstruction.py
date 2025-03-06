import os
import time
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt

###############################################################################
# 1) DATA PREPROCESSING (Using OTU columns that start with "d__")
###############################################################################

def preprocess_single_sample(csv_path, otu_prefix="d__", num_otus=10):
    """
    Loads the CSV from `csv_path`, selects OTU columns that start with `otu_prefix`,
    picks the first `num_otus` OTU columns, and returns the first sample as a DataFrame.
    """
    print("Loading data from:", csv_path)
    df = pd.read_csv(csv_path)
    
    # Select OTU columns starting with otu_prefix
    otu_cols = [c for c in df.columns if c.startswith(otu_prefix)]
    if len(otu_cols) == 0:
        raise ValueError("No OTU columns found. Check your CSV or prefix.")
    
    # Limit to first num_otus OTU columns
    otu_cols = otu_cols[:num_otus]
    print(f"Selected {len(otu_cols)} OTU columns: {otu_cols}")
    
    # We'll consider only the first sample (row 0)
    sample_df = df.iloc[[0]].copy()
    sample_df = sample_df[otu_cols]
    
    # Convert OTU columns to numeric and fill NaNs
    for col in otu_cols:
        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(0)
    
    # Convert counts to relative abundance
    total = sample_df.sum(axis=1).iloc[0]
    if total == 0:
        total = 1e-9
    sample_df = sample_df.div(total, axis=0)
    
    # Apply double-square-root transform
    for col in otu_cols:
        sample_df[col] = np.sqrt(np.sqrt(sample_df[col].astype(float)))
    
    print("Preprocessing complete. Sample shape:", sample_df.shape)
    return sample_df, otu_cols

###############################################################################
# 2) BUILD GRAPH EDGES (For a single sample)
###############################################################################

def mantel_test(dist_matrix1, dist_matrix2, permutations=99):
    """
    Compute Mantel test correlation between two condensed distance vectors.
    (Reduced permutations for quick debugging.)
    Returns (r_obs, p_value).
    """
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

def build_graph_edges(sample_df, otu_cols, alpha=0.05):
    """
    For a single sample, build a graph where each node is one of the selected OTUs.
    Because there is only one sample, traditional pdist returns an empty array.
    For debugging, we'll create a fully connected graph among the OTU nodes with a constant weight.
    """
    features = otu_cols  # only OTU nodes
    num_nodes = len(features)
    edges = []
    weights = []
    # Fully connected graph (excluding self-loops)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edges.append((i, j))
            edges.append((j, i))
            weights.append(1.0)
            weights.append(1.0)
    
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    print(f"Constructed graph with {num_nodes} nodes and {edge_index.shape[1]} edges.")
    return features, edge_index, edge_weight

###############################################################################
# 3) VISUALIZE STATIC GRAPH
###############################################################################

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
    plt.title("Static Graph Topology (10 OTU Nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved static graph topology to {save_path}")

###############################################################################
# 4) CREATE A SINGLE GRAPH (DEBUG)
###############################################################################

def create_single_graph(sample_df, feature_list, edge_index, edge_weight):
    """
    Creates one PyG Data object for the single sample.
    """
    x_vals = sample_df[feature_list].iloc[0].to_numpy(dtype=float)
    x_tensor = torch.tensor(x_vals, dtype=torch.float).view(-1,1)
    data = Data(x=x_tensor, edge_index=edge_index.clone(), edge_weight=edge_weight.clone())
    return data

def visualize_sample_graph(data, feature_list, save_path="sample_graph_debug.png"):
    """
    Visualize a single sample graph with proper colorbar setup
    """
    G = nx.Graph()
    n_nodes = data.x.shape[0]
    G.add_nodes_from(range(n_nodes))
    if data.edge_index.shape[1] > 0:
        edge_list = data.edge_index.t().tolist()
        G.add_edges_from(edge_list)
    
    node_values = data.x.view(-1).cpu().numpy()
    labels = {i: feature_list[i] for i in range(n_nodes)}
    
    # Create figure and axis objects with a single subplot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_values, 
                                 cmap=plt.cm.viridis, 
                                 node_size=600,
                                 ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add colorbar using the nodes mappable
    plt.colorbar(nodes, ax=ax, label="Node Feature Value")
    
    plt.title("Sample Graph (10 OTU Nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved sample graph visualization to {save_path}")

###############################################################################
# MAIN EXECUTION (DEBUG MODE: ONE SAMPLE, 10 OTUs)
###############################################################################

def main():
    csv_path = "Data/New_data.csv"  # Update path if needed
    print("Preprocessing single sample with 10 OTUs...")
    sample_df, selected_otus = preprocess_single_sample(csv_path, otu_prefix="d__", num_otus=10)
    print("Selected OTUs:", selected_otus)
    
    print("Building graph edges for the single sample...")
    feat_list, edge_index, edge_weight = build_graph_edges(sample_df, selected_otus, alpha=0.05)
    visualize_graph_static(feat_list, edge_index, save_path="graph_topology.png")
    
    print("Creating single graph data object...")
    data = create_single_graph(sample_df, feat_list, edge_index, edge_weight)
    visualize_sample_graph(data, feat_list, save_path="sample_graph_debug.png")
    print("Graph creation debug run complete.")

if __name__ == "__main__":
    main()
