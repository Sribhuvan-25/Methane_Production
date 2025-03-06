import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

# Dummy implementation for mantel_test (replace with your actual implementation)
def mantel_test(dist_vec1, dist_vec2, permutations=999):
    # Here, we simply compute Pearson correlation as a placeholder.
    # In practice, you should replace this with a proper Mantel test implementation.
    r_val, p_val = pearsonr(dist_vec1, dist_vec2)
    return r_val, p_val

def setup_gpu():
    """Configure GPU settings and return device"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

def calculate_static_edges(df: pd.DataFrame, taxa_cols: List[str], 
                           target_cols: List[str], alpha: float = 0.05) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Calculate static edge structure using Mantel test.
    This represents the invariant network topology based on the entire dataset.
    """
    # Combine taxa and additional features
    features = taxa_cols + target_cols
    dist_vectors = {}
    
    # Calculate distance vectors for each feature
    for feat in features:
        vals = df[feat].to_numpy().reshape(-1, 1)
        metric = "braycurtis" if feat in taxa_cols else "euclidean"
        dist_vectors[feat] = pdist(vals, metric=metric)

    edges = []
    weights = []
    
    # Perform Mantel test for each pair of features
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            r_val, p_val = mantel_test(dist_vectors[features[i]], 
                                       dist_vectors[features[j]], 
                                       permutations=999)
            if p_val < alpha:
                # add both directions for an undirected graph
                edges.extend([(i, j), (j, i)])
                weights.extend([r_val, r_val])

    if not edges:
        raise ValueError("No significant edges found. Adjust thresholds or check data.")

    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return features, edge_index, edge_weight

def create_graphs(df: pd.DataFrame, feature_list: List[str], 
                  edge_index: torch.Tensor, edge_weight: torch.Tensor,
                  target_cols: List[str], device: torch.device) -> List[Data]:
    """
    Create a graph for each sample.
    Each graph maintains the static edge structure (from the entire dataset)
    but has node features specific to that sample.
    """
    dataset = []
    target_indices = [feature_list.index(tc) for tc in target_cols]
    
    # Move static structures to GPU if available
    if device.type == 'cuda':
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    
    # Create one graph per sample (each row in the dataframe)
    for idx, row in df.iterrows():
        # Get feature values for this sample as a numpy array
        x_vals = row[feature_list].to_numpy(dtype=np.float32)
        # Optionally, mask out the target features if you want the model to predict them
        for ti in target_indices:
            x_vals[ti] = 0.0
            
        # Create tensor for node features; each node is a feature
        x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
        # Set the target (output) values separately
        y_tensor = torch.tensor(row[target_cols].to_numpy(dtype=np.float32))
        
        if device.type == 'cuda':
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)
            
        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y_tensor
        )
        dataset.append(data)
    
    print(f"Created {len(dataset)} graphs (one per sample)")
    return dataset

def main():
    device = setup_gpu()
    csv_path = "Data/New_data.csv"
    
    # Load and preprocess data using your preprocessing function.
    # This function should filter the OTUs based on abundance and prevalence thresholds,
    # and separate the taxa (OTU) columns from the additional target columns (e.g., ACE-km, H2-km, Biomass, Digester, etc.)
    from GCN.graph_generation import preprocess_data
    df_filtered, taxa_cols, target_cols = preprocess_data(
        csv_path,
        abundance_threshold=0.001,
        prevalence_threshold=0.01,
        otu_prefix="d__"
    )
    
    # Calculate static edge structure over the entire dataset
    feature_list, edge_index, edge_weight = calculate_static_edges(
        df_filtered, taxa_cols, target_cols, alpha=0.05
    )
    
    # Create a graph for each sample (no temporal windows, as data is static)
    dataset = create_graphs(
        df_filtered,
        feature_list,
        edge_index,
        edge_weight,
        target_cols,
        device
    )
    
    # Save the dataset to a file
    dataset_cpu = [data.cpu() for data in dataset]
    torch.save((dataset_cpu, feature_list, target_cols), "graph_dataset_static.pt")
    print("Saved static graph dataset to graph_dataset_static.pt")

if __name__ == "__main__":
    main()
