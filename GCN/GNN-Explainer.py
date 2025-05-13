#%% Imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GIN
from torch_geometric.explain import GNNExplainer, Explainer
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import itertools
import networkx as nx
from matplotlib.colors import Normalize

#%% Load and preprocess data (reusing from Embeddings.py)
def load_and_preprocess_data():
    data_path = "../Data/New_data.csv"
    df = pd.read_csv(data_path)
    df = df[~df.isin(['x']).any(axis=1)]
    
    # Convert target columns to numeric, replacing any non-numeric values with NaN
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

#%% Build initial graph
def build_initial_graph(df_fam_rel, param_df):
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
        if p < 0.05:
            edge_i += [i, j]
            edge_j += [j, i]
    
    edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
    
    # Create graph data objects
    graphs = []
    for idx in range(len(feature_matrix)):
        x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)
        # Convert target values to float32 before creating tensor
        y_values = target_data.iloc[idx].values.astype(np.float32)
        y = torch.tensor(y_values, dtype=torch.float).view(1, 2)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return graphs, node_feature_names

#%% GNN Model
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.3):
        super().__init__()
        self.gnn = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout
        )
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.head(x)

#%% Model Wrapper for Explainer
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x, edge_index, batch=None):
        # Create a Data-like object that your model expects
        class DataContainer:
            pass
        
        data = DataContainer()
        data.x = x
        data.edge_index = edge_index
        data.batch = batch
        
        return self.model(data)

#%% Train model and get explanations
def train_and_explain(model, graphs, target_idx, target_name, device, node_feature_names):
    # Split data
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, 201):
        # Training
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
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y[:, target_idx:target_idx+1]
                val_loss += loss_fn(pred, target).item() * batch.num_graphs
        
        val_loss /= len(val_loader.dataset)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} Train MSE: {total_loss/len(train_loader.dataset):.4f} Val MSE: {val_loss:.4f}")
        
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
    
    # Wrap the model for the explainer
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Create the explainer
    algorithm = GNNExplainer(epochs=200)
    explainer = Explainer(
        model=wrapped_model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        )
    )
    
    # Get explanations for validation set
    edge_masks = []
    node_masks = []
    
    for data in val_graphs:
        data = data.to(device)
        # Create batch tensor if not present
        if data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        
        # Make tensors require grad for the explainer
        x = data.x.clone().detach().requires_grad_(True)
        edge_index = data.edge_index.clone().detach()
        
        # Generate explanation
        explanation = explainer(x, edge_index, batch=data.batch)
        edge_masks.append(explanation.edge_mask.cpu().numpy())
        node_masks.append(explanation.node_mask.cpu().numpy())
    
    return edge_masks, node_masks

#%% Create sparsified graph
def create_sparsified_graph(original_graph, edge_masks, sparsity=0.5):
    # Average edge masks across all explanations
    avg_edge_mask = np.mean(edge_masks, axis=0)
    
    # Get top edges based on importance
    num_edges = len(avg_edge_mask)
    num_keep = int(num_edges * (1 - sparsity))
    top_edge_indices = np.argsort(avg_edge_mask)[-num_keep:]
    
    # Create new edge index with only important edges
    new_edge_index = original_graph.edge_index[:, top_edge_indices]
    
    # Create new graph with sparsified edges
    sparsified_graph = Data(
        x=original_graph.x,
        edge_index=new_edge_index,
        y=original_graph.y
    )
    
    return sparsified_graph, avg_edge_mask

#%% Visualize graph
def visualize_graph(graph, node_feature_names, title, edge_weights=None, save_path=None):
    """
    Visualizes a PyTorch Geometric graph using networkx with improved styling.
    Args:
        graph: PyTorch Geometric Data object
        node_feature_names: List of node labels
        title: Plot title
        edge_weights: Optional array of edge weights
        save_path: Optional path to save the figure
    """
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes with type information
    for i, name in enumerate(node_feature_names):
        G.add_node(name, type='microbe')
    
    # Add edges with weights if provided
    edge_index = graph.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src = node_feature_names[edge_index[0, i]]
        dst = node_feature_names[edge_index[1, i]]
        if edge_weights is not None:
            G.add_edge(src, dst, weight=edge_weights[i])
        else:
            G.add_edge(src, dst)
    
    # Get all nodes
    microbe_nodes = list(G.nodes())
    
    # Use Kamada-Kawai layout for better visualization
    pos = nx.kamada_kawai_layout(G)
    
    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw edges
    if edge_weights is not None:
        weights = np.array(edge_weights)
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        edge_colors = plt.cm.viridis(weights)
        edges = nx.draw_networkx_edges(
            G, pos,
            width=[1 + 2*w for w in weights],
            edge_color=edge_colors,
            alpha=0.7,
            ax=ax
        )
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Edge Importance')
    else:
        nx.draw_networkx_edges(G, pos, edge_color='#888888', alpha=0.5, width=1, ax=ax)
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, nodelist=microbe_nodes, node_color='skyblue', node_shape='o', node_size=200, label='Microbial Family', ax=ax)
    labels_dict = {n: n for n in microbe_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.6), ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    ax.legend()
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df_fam_rel, param_df = load_and_preprocess_data()
    
    # Build initial graph
    print("Building initial graph...")
    graphs, node_feature_names = build_initial_graph(df_fam_rel, param_df)
    
    # Train model and get explanations for ACE-km
    print("\nTraining model and getting explanations for ACE-km...")
    model = GNNModel(
        in_channels=1,
        hidden_channels=64,
        num_layers=3,
        out_channels=1
    ).to(device)
    
    edge_masks, node_masks = train_and_explain(
        model, graphs, target_idx=0, target_name="ACE-km", 
        device=device, node_feature_names=node_feature_names
    )
    
    # Create sparsified graph
    print("\nCreating sparsified graph...")
    sparsified_graph, avg_edge_mask = create_sparsified_graph(graphs[0], edge_masks, sparsity=0.5)
    
    # Print statistics
    print(f"\nOriginal graph: {graphs[0].edge_index.shape[1]} edges")
    print(f"Sparsified graph: {sparsified_graph.edge_index.shape[1]} edges")
    
    
# %%
# Visualize edge importance distribution
# Histogram of averaged edge‑mask scores
plt.figure(figsize=(10, 5))
plt.hist(np.mean(edge_masks, axis=0), bins=50)
plt.title("Distribution of Edge Importance Scores")
plt.xlabel("Importance Score")
plt.ylabel("Frequency")
plt.savefig("edge_importance_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Original graph
print("\nVisualizing original graph...")
visualize_graph(graphs[0],
                node_feature_names,
                "Original Graph Structure",
                save_path="original_graph.png")

# Sparsified graph (50 % edges removed)
print("\nVisualizing sparsified graph...")
visualize_graph(sparsified_graph,
                node_feature_names,
                "Sparsified Graph Structure (50% edges removed)",
                edge_weights=avg_edge_mask,
                save_path="sparsified_graph.png")

# Persist the sparsified graph
print("\nSaving sparsified graph data...")
torch.save({
    'edge_index':          sparsified_graph.edge_index,
    'x':                   sparsified_graph.x,
    'y':                   sparsified_graph.y,
    'node_feature_names':  node_feature_names,
    'edge_importance':     avg_edge_mask
}, "sparsified_graph.pt")

print("\nDone! Files saved:")
print(" - original_graph.png")
print(" - sparsified_graph.png")
print(" - edge_importance_distribution.png")
print(" - sparsified_graph.pt")
# %%
