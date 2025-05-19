import os
import torch
import numpy as np
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import itertools
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """
    Load and preprocess data from CSV file.
    Returns:
        tuple: (df_fam_rel, param_df) - processed family data and parameters
    """
    data_path = "../Data/New_data.csv"
    df = pd.read_csv(data_path)
    df = df[~df.isin(['x']).any(axis=1)]
    
    # Convert target columns to numeric
    target_cols = ['ACE-km', 'H2-km']
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in target columns
    df = df.dropna(subset=target_cols)
    
    otu_cols = [c for c in df.columns if "d__" in c]
    meta_cols = [c for c in df.columns if c not in otu_cols]
    
    def extract_family_from_colname(colname):
        for part in colname.split(';'):
            part = part.strip()
            if part.startswith('f__'):
                return part[3:] or "UnclassifiedFamily"
        return "UnclassifiedFamily"
    
    col_to_family = {c: extract_family_from_colname(c) for c in otu_cols}
    family_to_cols = {}
    for c, fam in col_to_family.items():
        if fam not in family_to_cols:
            family_to_cols[fam] = []
        family_to_cols[fam].append(c)
    
    df_fam = pd.DataFrame({
        fam: df[cols].sum(axis=1)
        for fam, cols in family_to_cols.items()
    }, index=df.index)
    
    param_df = df[meta_cols].copy()
    df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
    
    return df_fam_rel, param_df

def build_initial_graph(df_fam_rel, param_df):
    """
    Build initial graph structure using Mantel Test for edge creation.
    Returns:
        tuple: (graphs, node_feature_names) - list of graph objects and node names
    """
    # Filter families
    presence_count = (df_fam_rel > 0).sum(0)
    prevalence = presence_count / df_fam_rel.shape[0]
    high_prev = prevalence[prevalence >= 0.05].index
    
    mean_abund = df_fam_rel.mean(0)
    high_abund = mean_abund[mean_abund >= 0.01].index
    
    selected_families = high_prev.intersection(high_abund)
    df_fam_rel_filtered = df_fam_rel[selected_families].copy()
    
    # Prepare features and targets
    target_cols = ['ACE-km', 'H2-km']
    target_data = param_df[target_cols].copy()
    df_microbe = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))
    node_feature_names = list(df_microbe.columns)
    num_nodes = len(node_feature_names)
    feature_matrix = df_microbe.values.astype(np.float32)
    
    # Build edges via Mantel Test
    def compute_distance_matrix(vec, metric='braycurtis'):
        dm = squareform(pdist(vec[:, None], metric=metric))
        return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    
    def mantel_test(d1, d2, permutations=999):
        n = d1.shape[0]
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
        if p < 0.07:
            edge_i += [i, j]
            edge_j += [j, i]
    
    edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
    
    # Create graph data objects
    graphs = []
    for idx in range(len(feature_matrix)):
        x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)
        y_values = target_data.iloc[idx].values.astype(np.float32)
        y = torch.tensor(y_values, dtype=torch.float).view(1, 2)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return graphs, node_feature_names

def local_degree_profiles(adjacency_matrix, radius):
    """
    Compute local degree profile features for nodes in the graph.
    
    Args:
        adjacency_matrix (numpy.ndarray): Input adjacency matrix
        radius (int): Radius for local degree profile computation
        
    Returns:
        numpy.ndarray: Node features matrix of shape (num_nodes, 5)
    """
    adj_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    ldp_list = []
    num_nodes = adj_matrix_tensor.shape[0]
    identity = torch.eye(num_nodes)
    
    node_degrees = []
    
    for node in range(num_nodes):
        node_neighbors = identity[node].clone()
        ldp = []
        
        # Calculate node degree
        node_degree = torch.sum(adj_matrix_tensor[node]).item()
        node_degrees.append(node_degree)
        
        for _ in range(radius):
            node_neighbors = torch.mm(adj_matrix_tensor, node_neighbors.unsqueeze(1)).squeeze(1)
            ldp.append(torch.sum(node_neighbors).item())
        
        ldp_list.append(ldp)
    mean_ldp = [torch.mean(torch.tensor(ldp)).item() for ldp in ldp_list]
    max_ldp = [torch.max(torch.tensor(ldp)).item() for ldp in ldp_list]
    min_ldp = [torch.min(torch.tensor(ldp)).item() for ldp in ldp_list]
    std_ldp = [torch.std(torch.tensor(ldp)).item() for ldp in ldp_list]
    combine = np.concatenate((node_degrees, mean_ldp, max_ldp, min_ldp, std_ldp))
    reshaped_combine = np.reshape(combine, (num_nodes, 5))
    return reshaped_combine

def sparsify_and_save_graph(adjacency_matrix, k_neighbors, output_path, compute_features=True, radius=2):
    """
    Sparsify a graph using KNN approach and save it in a format compatible with Embeddings.py.
    
    Args:
        adjacency_matrix (numpy.ndarray): Input adjacency matrix
        k_neighbors (int): Number of neighbors to keep for each node
        output_path (str): Path to save the sparsified graph data
        compute_features (bool): Whether to compute local degree profile features
        radius (int): Radius for local degree profile computation if compute_features is True
        
    Returns:
        tuple: (sparsified_adjacency_matrix, node_features)
            - sparsified_adjacency_matrix: The KNN-sparsified adjacency matrix
            - node_features: Node features if compute_features is True, None otherwise
    """
    # Create a copy of the input matrix
    sparsified_matrix = adjacency_matrix.copy()
    n_nodes = sparsified_matrix.shape[0]
    
    # For each node, find its k-nearest neighbors based on the original edge weights
    for i in range(n_nodes):
        # Get the row of edge weights for this node
        node_weights = sparsified_matrix[i].copy()
        
        # Find indices of k largest values (excluding self)
        node_weights[i] = -float('inf')  # Exclude self
        top_k_indices = np.argsort(node_weights)[-k_neighbors:]
        
        # Create new row with only k-nearest neighbors
        new_row = np.zeros_like(node_weights)
        new_row[top_k_indices] = node_weights[top_k_indices]  # Keep original weights
        
        # Update the matrix
        sparsified_matrix[i] = new_row
    
    # Make the matrix symmetric (undirected graph)
    sparsified_matrix = np.maximum(sparsified_matrix, sparsified_matrix.T)
    
    # Compute node features if requested
    node_features = None
    if compute_features:
        node_features = local_degree_profiles(adjacency_matrix, radius)
    
    # Convert to PyTorch Geometric Data format
    # Get indices and weights of non-zero elements
    rows, cols = np.where(sparsified_matrix > 0)
    weights = sparsified_matrix[rows, cols]
    
    # Create edge_index tensor
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    weights = weights[mask]
    
    # Sort edges to ensure consistent ordering
    sort_idx = edge_index[0].argsort()
    edge_index = edge_index[:, sort_idx]
    weights = weights[sort_idx]
    
    print(f"Original matrix shape: {adjacency_matrix.shape}")
    print(f"Number of edges in sparsified graph: {edge_index.shape[1]//2} (undirected)")
    print(f"Average edge weight: {weights.mean():.4f}")
    
    # Create graph data object
    graph_data = Data(
        x=torch.tensor(node_features).float() if node_features is not None else None,
        edge_index=edge_index,
        edge_attr=torch.tensor(weights, dtype=torch.float).view(-1, 1)  # Add edge weights
    )
    
    # Save the sparsified graph data
    torch.save({
        'edge_index': graph_data.edge_index,
        'edge_attr': graph_data.edge_attr,  # Save edge weights
        'x': graph_data.x if hasattr(graph_data, 'x') else None,
        'num_nodes': n_nodes,
        'k_neighbors': k_neighbors,
        'original_matrix_shape': adjacency_matrix.shape
    }, output_path)
    
    print(f"Sparsified graph saved to: {output_path}")
    
    return sparsified_matrix, node_features

def visualize_graphs(original_matrix, sparsified_matrix, node_names, k_neighbors, output_dir='graph_visualizations'):
    """
    Visualize both original and sparsified graphs with their properties and node names.
    
    Args:
        original_matrix (numpy.ndarray): Original adjacency matrix
        sparsified_matrix (numpy.ndarray): Sparsified adjacency matrix
        node_names (list): List of node names (e.g., family names)
        k_neighbors (int): The k value used for sparsification
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create NetworkX graphs
    G_original = nx.from_numpy_array(original_matrix)
    G_sparsified = nx.from_numpy_array(sparsified_matrix)
    
    # Calculate graph properties
    original_nodes = G_original.number_of_nodes()
    original_edges = G_original.number_of_edges()
    sparsified_nodes = G_sparsified.number_of_nodes()
    sparsified_edges = G_sparsified.number_of_edges()
    
    # Create node labels mapping
    labels = {i: name for i, name in enumerate(node_names)}
    
    # Create figure with two subplots
    plt.figure(figsize=(20, 10))
    
    # Plot original graph
    plt.subplot(121)
    pos = nx.spring_layout(G_original, seed=42)
    nx.draw(G_original, pos, with_labels=True, labels=labels, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold', edge_color='#cccccc')
    plt.title(f'Original Graph\nNodes: {original_nodes}, Edges: {original_edges}')
    
    # Plot sparsified graph
    plt.subplot(122)
    pos = nx.spring_layout(G_sparsified, seed=42)
    nx.draw(G_sparsified, pos, with_labels=True, labels=labels, node_color='lightgreen',
            node_size=500, font_size=8, font_weight='bold', edge_color='#cccccc')
    plt.title(f'Sparsified Graph\nNodes: {sparsified_nodes}, Edges: {sparsified_edges}')
    
    # Save the visualization with k in the filename
    plt.savefig(os.path.join(output_dir, f'graph_comparison_k{k_neighbors}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Original Graph: {original_nodes} nodes, {original_edges} edges")
    print(f"Sparsified Graph: {sparsified_nodes} nodes, {sparsified_edges} edges")
    if original_edges > 0:
        print(f"Edge reduction: {((original_edges - sparsified_edges) / original_edges * 100):.2f}%")
    else:
        print("Edge reduction: N/A (no edges in original graph)")

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df_fam_rel, param_df = load_and_preprocess_data()
    
    # Build initial graph
    print("Building initial graph...")
    graphs, node_feature_names = build_initial_graph(df_fam_rel, param_df)
    
    if not graphs:
        print("No graph data could be built. Exiting.")
        exit()
    
    # Convert the first graph's edge_index to adjacency matrix
    edge_index = graphs[0].edge_index.numpy()
    num_nodes = graphs[0].num_nodes
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    
    # Set parameters for sparsification
    k_neighbors = 5
    output_path = f'Sparsified_Graphs/sparsified_graph_k{k_neighbors}.pt'
    
    # Run sparsification
    print(f"\nSparsifying graph with k={k_neighbors}...")
    sparsified_matrix, node_features = sparsify_and_save_graph(
        adjacency_matrix=adjacency_matrix,
        k_neighbors=k_neighbors,
        output_path=output_path,
        compute_features=True,
        radius=2
    )
    
    # Visualize both graphs
    print("\nGenerating graph visualizations...")
    visualize_graphs(adjacency_matrix, sparsified_matrix, node_feature_names, k_neighbors)