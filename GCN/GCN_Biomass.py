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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import itertools
import random
import os

print("Running GCN with Biomass Features as Graph-Level Attributes")

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data Loading and Preprocessing
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)
print(f"Initial dataset size: {len(df)}")

# Remove rows with 'x' values
df = df[~df.isin(['x']).any(axis=1)]
print(f"Dataset size after removing rows with 'x': {len(df)}")

# if 'ACE-km' in df.columns:
#     print(f"Before outlier removal: {len(df)} samples")
#     df['ACE-km'] = pd.to_numeric(df['ACE-km'], errors='coerce')
#     df = df[df['ACE-km'] <= 33]
#     print(f"After removing ACE-km outliers > 33: {len(df)} samples")
# else:
#     print("Warning: 'ACE-km' column not found in the dataset")

# Extract OTU and metadata columns
otu_cols = [col for col in df.columns if "d__" in col]
meta_cols = [col for col in df.columns if col not in otu_cols]
print(f"Found {len(otu_cols)} OTU columns and {len(meta_cols)} metadata columns")

# Function to extract family information from column names
def extract_family_from_colname(colname):
    parts = colname.split(';')
    for part in parts:
        part = part.strip()
        if part.startswith('f__'):
            fam = part[3:]
            return fam if fam else "UnclassifiedFamily"
    return "UnclassifiedFamily"

# Map OTUs to families
col_to_family = {}
for c in otu_cols:
    family = extract_family_from_colname(c)
    col_to_family[c] = family

family_to_cols = collections.defaultdict(list)
for c in otu_cols:
    fam = col_to_family[c]
    family_to_cols[fam].append(c)

# Aggregate OTUs by family
df_fam = pd.DataFrame(index=df.index)
for fam, col_list in family_to_cols.items():
    df_fam[fam] = df[col_list].sum(axis=1)

param_df = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

# Save intermediate files
df_fam_rel.to_csv("family_abundance.csv", index=False)
param_df.to_csv("parameters.csv", index=False)

# Filtering Families by prevalence and abundance
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

# Constructing Node Features and extracting targets
target_cols = ['ACE-km', 'H2-km']

# Extract targets (graph-level labels)
target_data = param_df[target_cols].copy()

# Identify biomass type columns
# UPDATED: Use only specific biomass columns instead of searching
biomass_cols = ['Biomass_F', 'Biomass_G']
print(f"Using biomass columns: {biomass_cols}")

# Verify the columns exist in the dataset
missing_cols = [col for col in biomass_cols if col not in param_df.columns]
if missing_cols:
    print(f"WARNING: Some specified biomass columns not found in dataset: {missing_cols}")
    # Filter to only include columns that exist
    biomass_cols = [col for col in biomass_cols if col in param_df.columns]
    
    if not biomass_cols:
        print("ERROR: No specified biomass columns found in dataset. Creating dummy column.")
        param_df['DummyBiomass'] = 'Unknown'
        biomass_cols = ['DummyBiomass']

# Extract biomass features and convert to one-hot encoding if needed
biomass_features = param_df[biomass_cols].copy()
if biomass_features.select_dtypes(include=['object', 'category']).shape[1] > 0:
    biomass_features = pd.get_dummies(biomass_features)

print(f"Biomass type features after processing: {list(biomass_features.columns)}")
print(f"Biomass feature shape: {biomass_features.shape}")

# Remove target and biomass columns from other parameters
param_inputs = param_df.drop(columns=target_cols + biomass_cols, errors='ignore')

# Double square-root transformation for microbial relative abundances
df_microbe_sqrt = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))

# ONLY use microbial features as nodes
node_features_df = df_microbe_sqrt.copy()
print("Node feature matrix shape (samples x nodes):", node_features_df.shape)

# Record feature names
node_feature_names = list(node_features_df.columns)
print(f"Number of node features: {len(node_feature_names)}")

# Edge Construction via Mantel Test
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

# Compute distance matrices for each node (feature)
dist_mats = {}
for col in node_feature_names:
    values = node_features_df[col].values.astype(np.float32)
    # Replace any NaNs or infs with 0
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # All columns are microbial features, so use braycurtis distance
    dist_mats[col] = compute_distance_matrix(values, metric='braycurtis')

def mantel_test(dist_matrix1, dist_matrix2, permutations=999):
    n = dist_matrix1.shape[0]
    idx = np.triu_indices(n, k=1)
    d1 = dist_matrix1[idx]
    d2 = dist_matrix2[idx]
    if d1.std() == 0 or d2.std() == 0:
        return 1.0, 0.0
    r_obs = pearsonr(d1, d2)[0]
    count = 0
    for _ in range(permutations):
        perm = np.random.permutation(n)
        d2_perm = dist_matrix2[perm][:, perm]
        r_perm = pearsonr(d1, d2_perm[idx])[0]
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_val = (count + 1) / (permutations + 1)
    return p_val, r_obs

# Create edge index list
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
print(f"Total edges (undirected) defined by Mantel test: {edge_index.shape[1]//2}")

# Visualize the graph (optional)
def visualize_graph():
    G = nx.Graph()
    for i, col in enumerate(node_feature_names):
        G.add_node(col, type='microbe')
        
    for src, tgt in zip(edge_index_list[0], edge_index_list[1]):
        G.add_edge(node_feature_names[src], node_feature_names[tgt])
        
    microbe_nodes = list(G.nodes())
    
    pos = nx.kamada_kawai_layout(G)
    
    plt.figure(figsize=(12,12))
    nx.draw_networkx_edges(G, pos, edge_color='#888888', alpha=0.5, width=1)
    
    nx.draw_networkx_nodes(G, pos, nodelist=microbe_nodes, node_color='skyblue', 
                          node_shape='o', node_size=200, label='Microbial Family')
    
    labels_dict = {n: n for n in microbe_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.6))
    
    plt.title("Biogas Interaction Graph - Microbial Families Only")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig("graph_structure.png")
    plt.close()

visualize_graph()
print("Graph visualization saved to 'graph_structure.png'")

# Create PyG Data Objects for Each Sample
node_features_df = node_features_df.apply(pd.to_numeric, errors='coerce')
node_features = node_features_df.values.astype(np.float32)

# Ensure target data is numeric
target_data = target_data.apply(pd.to_numeric, errors='coerce')

# Ensure biomass features are numeric
biomass_features = biomass_features.astype(np.float32)

# Create graph objects with biomass features
graphs = []
for idx in range(len(node_features_df)):
    # Create node features tensor
    x = torch.tensor(node_features[idx], dtype=torch.float).unsqueeze(1)  # shape: [num_nodes, 1]
    
    # Create graph-level target tensor
    y_vals = target_data.iloc[idx].values.astype(np.float32)
    y = torch.tensor(y_vals, dtype=torch.float).unsqueeze(0)  # shape: [1, 2]
    
    # Create biomass features tensor
    biomass_vals = biomass_features.iloc[idx].values.astype(np.float32)
    biomass = torch.tensor(biomass_vals, dtype=torch.float)  # shape: [num_biomass_features]
    
    # Create graph with node features, edge_index, graph-level target, and biomass
    graph = Data(
        x=x, 
        edge_index=edge_index, 
        y=y,
        biomass=biomass  # Add biomass as graph attribute
    )
    graphs.append(graph)

print(f"Created {len(graphs)} graph data objects (one per sample).")
print(f"Each graph has {num_nodes} nodes (microbial families) and {edge_index.shape[1]//2} undirected edges.")
print(f"Target dimensions: {graphs[0].y.shape}")
print(f"Biomass feature dimensions: {graphs[0].biomass.shape}")

# Create GCN-Results directory for saving outputs
results_dir = "GCN-Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")
    
# Create subdirectories for different output types
plots_dir = os.path.join(results_dir, "plots")
metrics_dir = os.path.join(results_dir, "metrics")
models_dir = os.path.join(results_dir, "models")
samples_dir = os.path.join(results_dir, "sample_plots")

for directory in [plots_dir, metrics_dir, models_dir, samples_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Visualize sample node features
def visualize_sample(sample_idx, save_path=None):
    sample = graphs[sample_idx]
    
    # Extract node feature values from the sample
    sample_values = sample.x.squeeze().cpu().numpy()
    
    # Create a graph for visualization
    G = nx.Graph()
    for i, name in enumerate(node_feature_names):
        G.add_node(name, value=sample_values[i])
        
    # Add edges based on edge_index
    for i in range(0, edge_index.shape[1], 2):  # Only add one direction of edges
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        src_name = node_feature_names[src]
        tgt_name = node_feature_names[tgt]
        G.add_edge(src_name, tgt_name)
    
    # Set random seed for reproducible layout
    # Note: Some older versions of NetworkX don't accept 'seed' in kamada_kawai_layout
    np.random.seed(42)
    random.seed(42)
    
    # Use Kamada–Kawai layout for a balanced view (without specifying seed)
    pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes with color based on the sample's feature value
    plt.figure(figsize=(14, 14))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=[G.nodes[n]['value'] for n in G.nodes()],
        cmap=plt.cm.viridis,
        node_size=500
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Label ALL nodes, not just the top 15
    labels_dict = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8,
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.6))
    
    # Add information about graph-level targets for this sample
    target_values = sample.y.cpu().numpy()
    # Extract the target values properly
    ace_km = float(target_values[0, 0])
    h2_km = float(target_values[0, 1])
    
    # Add biomass information if available
    biomass_info = ""
    if hasattr(sample, 'biomass'):
        biomass_values = sample.biomass.cpu().numpy()
        for i, col in enumerate(biomass_features.columns):
            if biomass_values[i] > 0:
                biomass_info = f"\nBiomass Type: {col}"
    
    plt.title(f"Sample {sample_idx}: Family Abundances{biomass_info}\nACE-km = {ace_km:.2f}, H2-km = {h2_km:.2f}")
    plt.colorbar(nodes, label="Abundance (double sqrt transform)")
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Sample visualization saved to {save_path}")
    else:
        plt.show()

# Paper GCN model with Biomass Integration
class PaperGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, fc_hidden_dims=[128, 64, 32, 16], 
                 biomass_size=None, out_channels=2):
        super(PaperGCNModel, self).__init__()
        
        # 2 GCN layers as in the paper
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        # Process biomass if available
        self.has_biomass = biomass_size is not None and biomass_size > 0
        if self.has_biomass:
            self.biomass_fc = nn.Linear(biomass_size, fc_hidden_dims[2])
            # Adjust the third FC layer to accept combined features
            self.fc1 = nn.Linear(hidden_channels, fc_hidden_dims[0])
            self.fc2 = nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1])
            self.fc3 = nn.Linear(fc_hidden_dims[1] + fc_hidden_dims[2], fc_hidden_dims[2])  # Combined features
            self.fc4 = nn.Linear(fc_hidden_dims[2], fc_hidden_dims[3])
        else:
            # Standard FC layers
            self.fc1 = nn.Linear(hidden_channels, fc_hidden_dims[0])
            self.fc2 = nn.Linear(fc_hidden_dims[0], fc_hidden_dims[1])
            self.fc3 = nn.Linear(fc_hidden_dims[1], fc_hidden_dims[2])
            self.fc4 = nn.Linear(fc_hidden_dims[2], fc_hidden_dims[3])
        
        # Output layer
        self.out = nn.Linear(fc_hidden_dims[3], out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Safely handle batch information
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        # Global pooling to get graph-level features
        x = global_mean_pool(x, batch)
        
        # First two FC layers (standard)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        graph_feat = self.dropout(x)
        
        # Incorporate biomass features if available (at the third FC layer)
        if self.has_biomass and hasattr(data, 'biomass'):
            # Process biomass features
            if hasattr(data, 'batch') and data.batch is not None:
                # In batch mode, determine number of graphs
                num_graphs = int(batch.max()) + 1
                
                # Handle batched biomass features
                if data.biomass.dim() == 2 and data.biomass.size(0) == num_graphs:
                    biomass_batch = data.biomass
                else:
                    # Reshape to correct dimensions
                    biomass_batch = data.biomass.view(num_graphs, -1)
            else:
                # Single graph mode
                biomass_batch = data.biomass.unsqueeze(0) if data.biomass.dim() == 1 else data.biomass
            
            # Process biomass features
            biomass_feat = F.relu(self.biomass_fc(biomass_batch))
            biomass_feat = self.dropout(biomass_feat)
            
            # Combine features at the third FC layer
            combined_feat = torch.cat([graph_feat, biomass_feat], dim=1)
            x = F.relu(self.fc3(combined_feat))
        else:
            # Standard processing
            x = F.relu(self.fc3(graph_feat))
        
        x = self.dropout(x)
        
        # Fourth FC layer
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        # Output layer
        out = self.out(x)
        
        return out

# Utility functions for training and evaluation
def train_model(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss

def evaluate_model(model, data_loader, device):
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.append(pred.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    
    return preds, trues

def plot_results(all_trues, all_preds, model_name, target_names=['ACE-km', 'H2-km']):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(all_trues[:, i], all_preds[:, i], alpha=0.7, 
                   color='blue' if i == 0 else 'green', edgecolors='k')
        
        min_val = min(all_trues[:, i].min(), all_preds[:, i].min())
        max_val = max(all_trues[:, i].max(), all_preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal fit')
        
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"Predicted vs Actual: {name}")
        ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{model_name}_results.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Results plot saved to '{plot_path}'")

# 5-Fold Cross Validation
def run_cross_validation(model_class, model_name, graphs, n_epochs=101, model_params=None):
    if model_params is None:
        model_params = {}
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_fold_preds = []
    all_fold_trues = []
    
    # Create fold-specific directory
    fold_dir = os.path.join(results_dir, model_name, "folds")
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    
    for train_idx, val_idx in kf.split(graphs):
        print(f"\nStarting Fold {fold} for {model_name}")
        
        # Split data
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
        
        # Initialize model
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        
        # Training loop
        train_losses = []
        for epoch in range(1, n_epochs):
            total_loss = train_model(model, train_loader, optimizer, loss_fn, device)
            avg_loss = total_loss/len(train_graphs)
            train_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Fold {fold} Epoch {epoch:03d} | Train Loss: {avg_loss:.4f}")
        
        # Save training loss plot
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, n_epochs), train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'{model_name} - Fold {fold} Training Loss')
        loss_plot_path = os.path.join(fold_dir, f"fold{fold}_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Save trained model
        model_path = os.path.join(models_dir, f"{model_name}_fold{fold}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluation
        fold_preds, fold_trues = evaluate_model(model, val_loader, device)
        all_fold_preds.append(fold_preds)
        all_fold_trues.append(fold_trues)
        
        # Save fold-specific results
        fold_mse_ace = mean_squared_error(fold_trues[:, 0], fold_preds[:, 0])
        fold_mse_h2 = mean_squared_error(fold_trues[:, 1], fold_preds[:, 1])
        fold_r2_ace = r2_score(fold_trues[:, 0], fold_preds[:, 0])
        fold_r2_h2 = r2_score(fold_trues[:, 1], fold_preds[:, 1])
        
        fold_results = {
            'Fold': fold,
            'MSE_ACE': fold_mse_ace,
            'MSE_H2': fold_mse_h2,
            'R2_ACE': fold_r2_ace,
            'R2_H2': fold_r2_h2
        }
        
        fold_df = pd.DataFrame([fold_results])
        fold_csv_path = os.path.join(fold_dir, f"fold{fold}_metrics.csv")
        fold_df.to_csv(fold_csv_path, index=False)
        
        print(f"Fold {fold} completed. Validation samples: {fold_preds.shape[0]}")
        print(f"Fold {fold} MSE: ACE-km={fold_mse_ace:.4f}, H2-km={fold_mse_h2:.4f}")
        print(f"Fold {fold} R²: ACE-km={fold_r2_ace:.3f}, H2-km={fold_r2_h2:.3f}")
        fold += 1
    
    # Combine results from all folds
    all_preds = np.vstack(all_fold_preds)
    all_trues = np.vstack(all_fold_trues)
    
    # Save combined predictions
    pred_df = pd.DataFrame(all_preds, columns=['ACE_pred', 'H2_pred'])
    true_df = pd.DataFrame(all_trues, columns=['ACE_true', 'H2_true'])
    results_combined = pd.concat([true_df, pred_df], axis=1)
    results_path = os.path.join(metrics_dir, f"{model_name}_predictions.csv")
    results_combined.to_csv(results_path, index=False)
    
    # Calculate metrics
    mse_ace = mean_squared_error(all_trues[:, 0], all_preds[:, 0])
    mse_h2 = mean_squared_error(all_trues[:, 1], all_preds[:, 1])
    r2_ace = r2_score(all_trues[:, 0], all_preds[:, 0])
    r2_h2 = r2_score(all_trues[:, 1], all_preds[:, 1])
    
    print(f"\nOverall CV MSE: ACE-km={mse_ace:.4f}, H2-km={mse_h2:.4f}")
    print(f"Overall CV R^2: ACE-km={r2_ace:.3f}, H2-km={r2_h2:.3f}")
    
    # Plot results
    plot_results(all_trues, all_preds, model_name)
    
    return {
        'mse': {'ACE-km': mse_ace, 'H2-km': mse_h2},
        'r2': {'ACE-km': r2_ace, 'H2-km': r2_h2},
        'predictions': all_preds,
        'true_values': all_trues
    }

# Main execution block
if __name__ == "__main__":
    # Get biomass size
    biomass_size = biomass_features.shape[1]
    print(f"Biomass feature size: {biomass_size}")
    
    # Save the graph structure image to the plots directory
    plt.figure(figsize=(12,12))
    graph_vis_path = os.path.join(plots_dir, "graph_structure.png")
    # Copy the graph visualization if it exists
    if os.path.exists("graph_structure.png"):
        import shutil
        shutil.copy("graph_structure.png", graph_vis_path)
        print(f"Graph structure visualization saved to {graph_vis_path}")
    
    # Save node features and biomass features distribution
    plt.figure(figsize=(10, 6))
    node_features_array = node_features_df.values
    plt.hist(node_features_array.flatten(), bins=50, alpha=0.7)
    plt.title("Node Features Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    node_feat_path = os.path.join(plots_dir, "node_features_distribution.png")
    plt.savefig(node_feat_path)
    plt.close()
    print(f"Node features distribution saved to {node_feat_path}")
    
    # Save biomass features distribution
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(biomass_features.columns):
        plt.hist(biomass_features[col].values, bins=20, alpha=0.7, label=col)
    plt.title("Biomass Features Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    biomass_feat_path = os.path.join(plots_dir, "biomass_features_distribution.png")
    plt.savefig(biomass_feat_path)
    plt.close()
    print(f"Biomass features distribution saved to {biomass_feat_path}")
    
    # Visualize some sample graphs
    print("\nVisualizing sample graphs...")
    sample_indices = [0, 10, 20, 30, 40]  # Choose a few sample indices to visualize
    for idx in sample_indices:
        if idx < len(graphs):
            sample_path = os.path.join(samples_dir, f"sample_{idx}.png")
            visualize_sample(idx, save_path=sample_path)
    
    # Run cross-validation with the PaperGCNModel
    print("\nRunning cross-validation with PaperGCNModel...")
    model_params = {
        'in_channels': 1,  # Each node has 1 feature (abundance)
        'hidden_channels': 64,
        'fc_hidden_dims': [128, 64, 32, 16],
        'biomass_size': biomass_size,
        'out_channels': 2  # ACE-km and H2-km
    }
    
    results = run_cross_validation(
        PaperGCNModel, 
        "PaperGCNModel",
        graphs, 
        n_epochs=101,
        model_params=model_params
    )
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'R²'],
        'ACE-km': [results['mse']['ACE-km'], results['r2']['ACE-km']],
        'H2-km': [results['mse']['H2-km'], results['r2']['H2-km']]
    })
    
    metrics_path = os.path.join(metrics_dir, "papergcn_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to '{metrics_path}'")
    
    print("\nGCN with Biomass Features Pipeline Completed Successfully!") 