import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

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

def create_time_windows(df: pd.DataFrame, window_size: int = 3) -> List[pd.DataFrame]:
    """
    Create overlapping time windows from the dataset
    Each window represents a temporal snapshot of the microbial community
    """
    windows = []
    for i in range(len(df) - window_size + 1):
        windows.append(df.iloc[i:i+window_size])
    return windows

def calculate_static_edges(df: pd.DataFrame, taxa_cols: List[str], 
                         target_cols: List[str], alpha: float = 0.05) -> Tuple:
    """
    Calculate static edge structure using Mantel test
    This represents the invariant network topology
    """
    features = taxa_cols + target_cols
    dist_vectors = {}
    
    # Calculate distance matrices
    for feat in features:
        vals = df[feat].to_numpy().reshape(-1, 1)
        metric = "braycurtis" if feat in taxa_cols else "euclidean"
        dist_vectors[feat] = pdist(vals, metric=metric)

    edges = []
    weights = []
    
    # Perform Mantel test for each pair
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            r_val, p_val = mantel_test(dist_vectors[features[i]], 
                                     dist_vectors[features[j]], 
                                     permutations=999)
            if p_val < alpha:
                edges.extend([(i, j), (j, i)])
                weights.extend([r_val, r_val])

    if not edges:
        raise ValueError("No significant edges found. Adjust thresholds or check data.")

    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return features, edge_index, edge_weight

def create_temporal_graphs(df: pd.DataFrame, feature_list: List[str], 
                         edge_index: torch.Tensor, edge_weight: torch.Tensor,
                         target_cols: List[str], device: torch.device,
                         window_size: int = 3) -> List[Data]:
    """
    Create multiple graphs per sample using temporal windows
    Each graph maintains the static edge structure but has different node features
    based on temporal patterns
    """
    dataset = []
    target_indices = [feature_list.index(tc) for tc in target_cols]
    windows = create_time_windows(df, window_size)
    
    # Move static structures to GPU if available
    if device.type == 'cuda':
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    
    for window in windows:
        # Calculate temporal features
        for idx, row in window.iterrows():
            # Base features
            x_vals = row[feature_list].to_numpy(dtype=np.float32)
            
            # Mask target nodes
            for ti in target_indices:
                x_vals[ti] = 0.0
                
            # Create tensors
            x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
            y_tensor = torch.tensor(row[target_cols].to_numpy(dtype=np.float32))
            
            # Move to GPU if available
            if device.type == 'cuda':
                x_tensor = x_tensor.to(device)
                y_tensor = y_tensor.to(device)
            
            # Create graph with temporal context
            data = Data(
                x=x_tensor,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y_tensor
            )
            dataset.append(data)
    
    print(f"Created {len(dataset)} graphs using temporal windows")
    return dataset

def main():
    device = setup_gpu()
    csv_path = "Data/New_data.csv"
    
    # Load and preprocess data using the original preprocessing function
    from GCN.graph_generation import preprocess_data
    df_filtered, taxa_cols, target_cols = preprocess_data(
        csv_path,
        abundance_threshold=0.001,
        prevalence_threshold=0.01,
        otu_prefix="d__"
    )
    
    # Calculate static edge structure
    feat_list, edge_index, edge_weight = calculate_static_edges(
        df_filtered, taxa_cols, target_cols, alpha=0.05
    )
    
    # Create multiple graphs using temporal windows
    dataset = create_temporal_graphs(
        df_filtered,
        feat_list,
        edge_index,
        edge_weight,
        target_cols,
        device,
        window_size=3  # Adjust based on your temporal analysis needs
    )
    
    # Save the dataset
    dataset_cpu = [(data.cpu() if hasattr(data, 'cpu') else data) for data in dataset]
    torch.save((dataset_cpu, feat_list, target_cols), "graph_dataset_temporal.pt")
    print("Saved temporal dataset to graph_dataset_temporal.pt")

if __name__ == "__main__":
    main() 