import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt

def preprocess_data(csv_path, abundance_threshold=0.01, prevalence_threshold=0.05, otu_prefix="d__"):
    print("Loading data from:", csv_path)
    df = pd.read_csv(csv_path)

    target_cols = ["ACE-km", "H2-km"]
    all_cols = df.columns.tolist()
    otu_cols = [c for c in all_cols if c.startswith(otu_prefix)]
    keep_cols = otu_cols + target_cols

    # Remove rows with non-numeric or "x" contamination
    for col in keep_cols:
        df = df[df[col].astype(str).str.contains('x', case=False) == False]

    # Keep only OTU columns + targets
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

    # Convert OTU counts to relative abundance
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
        print("WARNING: No OTUs remain after filtering! Only targets remain.")

    # Double-square-root transform for OTUs
    for col in keep_otus:
        df_filtered[col] = np.sqrt(np.sqrt(df_filtered[col].astype(float)))
        # Optionally standardize
        mean_val = df_filtered[col].mean()
        std_val = df_filtered[col].std(ddof=0)
        if std_val != 0:
            df_filtered[col] = (df_filtered[col] - mean_val) / std_val

    print("Preprocessing complete. Data shape:", df_filtered.shape)
    return df_filtered, keep_otus, target_cols

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
    edge_list = edge_index.t().tolist() if edge_index.shape[1] > 0 else []
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

def create_dataset(df, feature_list, edge_index, edge_weight, target_cols):
    from torch_geometric.data import Data
    dataset = []
    target_indices = [feature_list.index(tc) for tc in target_cols]
    
    for idx, row in df.iterrows():
        x_vals = row[feature_list].to_numpy(dtype=float)
        # Mask out target nodes in input
        for ti in target_indices:
            x_vals[ti] = 0.0
        x_tensor = torch.tensor(x_vals, dtype=torch.float).view(-1,1)
        y_vals = row[target_cols].to_numpy(dtype=float)
        y_tensor = torch.tensor(y_vals, dtype=torch.float)
        data = Data(
            x=x_tensor,
            edge_index=edge_index.clone(),
            edge_weight=edge_weight.clone(),
            y=y_tensor
        )
        dataset.append(data)
    print(f"Created {len(dataset)} graphs (one per sample).")
    return dataset

def main():
    csv_path = "Data/New_data.csv"
    df_filtered, taxa_cols, target_cols = preprocess_data(
        csv_path,
        abundance_threshold=0.001,
        prevalence_threshold=0.01,
        otu_prefix="d__"
    )
    feat_list, edge_index, edge_weight = build_graph_edges(
        df_filtered, taxa_cols, target_cols, alpha=0.05
    )
    visualize_graph_static(feat_list, edge_index, "graph_topology.png")
    
    dataset = create_dataset(df_filtered, feat_list, edge_index, edge_weight, target_cols)
    
    # Save the dataset so the training file can load it
    torch.save((dataset, feat_list, target_cols), "graph_dataset.pt")
    print("Saved dataset to graph_dataset.pt")

if __name__ == "__main__":
    main()
