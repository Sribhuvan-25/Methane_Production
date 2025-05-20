#%%
import os
import collections
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from torch_geometric.utils import remove_self_loops, coalesce
import networkx as nx

#%% ========== Data Loading and Preprocessing ==========
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
param_df = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
# Filtering Families
presence_count = (df_fam_rel > 0).sum(0)
prevalence     = presence_count / df_fam_rel.shape[0]
high_prev      = prevalence[prevalence >= 0.05].index
mean_abund     = df_fam_rel.mean(0)
high_abund     = mean_abund[mean_abund >= 0.01].index
selected_families = high_prev.intersection(high_abund)
df_fam_rel_filtered = df_fam_rel[selected_families].copy()
# Ensure all values are numeric and handle non-numeric values
df_fam_rel_filtered = df_fam_rel_filtered.apply(pd.to_numeric, errors='coerce')
df_fam_rel_filtered = df_fam_rel_filtered.fillna(0)
# Node features & targets
node_feature_names = list(df_fam_rel_filtered.columns)
num_nodes = len(node_feature_names)
feature_matrix = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x))).values.astype(np.float32)
target_cols = ['ACE-km', 'H2-km']
target_data = param_df[target_cols].copy()
target_data = target_data.apply(pd.to_numeric, errors='coerce')
target_data = target_data.fillna(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% ========== Diagnostics ==========
def diagnostics(graphs, node_feature_names, target_data):
    print("\n=== Diagnostics ===")
    print(f"Number of graphs: {len(graphs)}")
    print(f"Node feature names: {node_feature_names}")
    print(f"Target columns: {list(target_data.columns)}")
    x0 = graphs[0].x.squeeze().cpu().numpy()
    print(f"Sample 0 node features (min, max, mean): {x0.min():.4f}, {x0.max():.4f}, {x0.mean():.4f}")
    print(f"Sample 0 node features: {x0}")
    y0 = graphs[0].y.cpu().numpy()
    print(f"Sample 0 target: {y0}")
    print(f"Target min: {target_data.min().values}, max: {target_data.max().values}, mean: {target_data.mean().values}")
    if hasattr(graphs[0], 'edge_attr') and graphs[0].edge_attr is not None:
        edge_w = graphs[0].edge_attr.cpu().numpy()
        print(f"Edge weights (min, max, mean): {edge_w.min():.4f}, {edge_w.max():.4f}, {edge_w.mean():.4f}")
        print(f"First 10 edge weights: {edge_w[:10].ravel()}")
    print("====================\n")

#%% ========== Baseline Check ==========
def baseline_check(feature_matrix, target_data, node_feature_names, target_idx=0):
    y = target_data.values[:, target_idx]
    print("\n=== Baseline Regression and Feature-Target Correlation ===")
    transforms = {
        'raw': lambda x: x,
        'log1p': lambda x: np.log1p(x),
        'zscore': lambda x: (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8),
        'sqrt': lambda x: np.sqrt(x),
        'sqrt_sqrt': lambda x: np.sqrt(np.sqrt(x)),
    }
    for name, func in transforms.items():
        X = func(feature_matrix)
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        et = ExtraTreesRegressor(n_estimators=100, random_state=42)
        et.fit(X, y)
        y_pred_et = et.predict(X)
        mse_et = mean_squared_error(y, y_pred_et)
        r2_et = r2_score(y, y_pred_et)
        print(f"[{name}] LinearRegression: MSE={mse:.4f}, R2={r2:.4f} | ExtraTrees: MSE={mse_et:.4f}, R2={r2_et:.4f}")
    X = transforms['sqrt_sqrt'](feature_matrix)
    plt.figure(figsize=(18, 6))
    for i in range(X.shape[1]):
        plt.subplot(1, min(5, X.shape[1]), i+1)
        plt.scatter(X[:, i], y, alpha=0.7)
        plt.xlabel(node_feature_names[i])
        plt.ylabel('Target')
        plt.title(f'Corr: {np.corrcoef(X[:, i], y)[0,1]:.2f}')
        if i == 4:
            break
    plt.suptitle('Feature-Target Correlations (first 5 features)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    corrs = [np.corrcoef(X[:, i], y)[0,1] for i in range(X.shape[1])]
    sorted_idx = np.argsort(np.abs(corrs))[::-1]
    print("Top 5 features by absolute correlation with target:")
    for idx in sorted_idx[:5]:
        print(f"{node_feature_names[idx]}: corr={corrs[idx]:.3f}")
    print("====================\n")

#%% ========== GNN Model ==========
class BuiltinGNN(nn.Module):
    def __init__(self, model_name, in_channels, hidden_channels, num_layers, out_channels, dropout=0.3, act='relu', jk=None, **kwargs):
        super().__init__()
        model_map = {
            'GCN': GCN,
            'GraphSAGE': GraphSAGE,
            'GIN': GIN,
            'GAT': GAT,
            'PNA': PNA,
        }
        if model_name not in model_map:
            raise ValueError(f"Unsupported model: {model_name}")
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
        head_in_dim = hidden_channels * num_layers if jk == 'cat' else hidden_channels
        self.head = nn.Linear(head_in_dim, out_channels)
    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.head(x)
    def get_graph_embeddings(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        x = global_mean_pool(x, data.batch)
        return x

#%% ========== Direct GNN Prediction ==========
def direct_gnn_prediction(graphs, target_data, target_idx=0, model_name='GCN', hidden_channels=64, num_layers=3, dropout=0.3, jk=None, **kwargs):
    print("\n=== Direct GNN Prediction ===")
    loader = DataLoader(graphs, batch_size=16, shuffle=False)
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
    loss_fn = nn.MSELoss()
    num_epochs = 100
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.y[:, target_idx:target_idx+1]
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        if epoch % 20 == 0:
            print(f"Epoch {epoch} Train MSE: {total_loss/len(loader.dataset):.4f}")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.append(pred.cpu().numpy())
            trues.append(batch.y[:, target_idx:target_idx+1].cpu().numpy())
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()
    mse = mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    print(f"Direct GNN prediction MSE: {mse:.4f}, R²: {r2:.4f}")
    print("First 10 predictions vs targets:")
    for i in range(min(10, len(preds))):
        print(f"Pred: {preds[i]:.3f}, Target: {trues[i]:.3f}")
    print("====================\n")

#%% ========== Sample-Specific KNN Graph Construction ==========
def build_sample_specific_knn_graphs(feature_matrix, target_data, node_feature_names, k=2):
    n_samples, n_nodes = feature_matrix.shape
    graphs = []
    for idx in range(n_samples):
        x = torch.tensor(feature_matrix[idx], dtype=torch.float).unsqueeze(1)
        abund = feature_matrix[idx]
        sim = 1.0 / (1.0 + np.abs(abund[:, None] - abund[None, :]))
        np.fill_diagonal(sim, 0)
        edge_set = set()
        edge_list = []
        for i in range(n_nodes):
            top_k = np.argsort(sim[i])[-k:]
            for j in top_k:
                if i != j and sim[i, j] > 0:
                    if (i, j) not in edge_set:
                        edge_set.add((i, j))
                        edge_list.append((i, j, sim[i, j]))
        if not edge_list:
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        edge_list.append((i, j, 1.0))
        edge_i, edge_j, edge_w = zip(*edge_list)
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_attr = torch.tensor(edge_w, dtype=torch.float).view(-1, 1)
        y = torch.tensor(target_data.iloc[idx].values, dtype=torch.float).view(1, 2)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs
#%% ========== Visualization ==========
def visualize_and_save_sample_graph(graph, node_feature_names, output_dir='sample_graph_visualizations', sample_idx=0):
    os.makedirs(output_dir, exist_ok=True)
    edge_index_np = graph.edge_index.cpu().numpy()
    G_nx = nx.DiGraph()
    G_nx.add_edges_from(zip(edge_index_np[0], edge_index_np[1]))
    labels = {i: name for i, name in enumerate(node_feature_names)}
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G_nx, seed=42)
    nx.draw(G_nx, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=8, font_weight='bold', edge_color='#cccccc', arrows=True)
    n_nodes = G_nx.number_of_nodes()
    n_edges = G_nx.number_of_edges()
    plt.title(f'Sample {sample_idx} Graph Visualization\nNodes: {n_nodes}, Edges: {n_edges}')
    plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample {sample_idx} graph visualization to {output_dir}/sample_{sample_idx}_graph.png")

#%% ========== Main Block ==========
if __name__ == "__main__":
    print("\n--- Baseline Check ---")
    baseline_check(feature_matrix, target_data, node_feature_names, target_idx=0)
    print("\n--- Sample-Specific KNN Graphs (Feature Similarity) ---")
    graphs_knn = build_sample_specific_knn_graphs(feature_matrix, target_data, node_feature_names, k=3)
    diagnostics(graphs_knn, node_feature_names, target_data)
    # Visualize and save the first sample's graph
    visualize_and_save_sample_graph(graphs_knn[0], node_feature_names, output_dir='sample_graph_visualizations', sample_idx=0)
    for model_name in ['GCN']: # 'GraphSAGE', 'GIN', 'GAT', 'PNA'
        print(f"\n=== Direct GNN Prediction with {model_name} ===")
        direct_gnn_prediction(
            graphs_knn, target_data, target_idx=0, model_name=model_name,
            hidden_channels=64, num_layers=3, dropout=0.3, jk=None
        ) 

