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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict

# Enable tensor cores for faster GPU computation if available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

###############################################################################
# 1) OPTIMIZED DATA PREPROCESSING
###############################################################################

def preprocess_data(csv_path: str, abundance_threshold: float = 0.001, 
                   prevalence_threshold: float = 0.01, otu_prefix: str = "d__") -> Tuple:
    """Optimized data preprocessing with better memory management"""
    print("Loading data from:", csv_path)
    
    # Read data without forcing dtypes initially
    df = pd.read_csv(csv_path)
    
    target_cols = ["ACE-km", "H2-km"]
    otu_cols = [c for c in df.columns if c.startswith(otu_prefix)]
    keep_cols = otu_cols + target_cols
    
    # Remove rows with 'x' values
    mask = ~df[keep_cols].astype(str).apply(lambda x: x.str.contains('x', case=False)).any(axis=1)
    df = df.loc[mask, keep_cols].copy()
    
    # Now convert to float32 after removing problematic rows
    df = df.astype('float32')
    
    # Vectorized operations for relative abundances
    otu_data = df[otu_cols]
    sums = otu_data.sum(axis=1)
    sums = sums.where(sums != 0, 1e-9)
    otu_data = otu_data.div(sums, axis=0)
    
    # Vectorized OTU filtering
    keep_otus = otu_data.columns[(otu_data > abundance_threshold).mean() >= prevalence_threshold].tolist()
    print(f"Filtered OTUs: {len(otu_cols)} -> {len(keep_otus)}")
    
    # Final processing
    final_cols = keep_otus + target_cols
    df_filtered = df[final_cols]
    
    # Vectorized transformation of OTU columns
    otu_data = df_filtered[keep_otus]
    otu_data = np.sqrt(np.sqrt(otu_data))
    
    # Vectorized standardization
    otu_stats = otu_data.agg(['mean', 'std'])
    otu_data = (otu_data - otu_stats.loc['mean']) / otu_stats.loc['std'].replace(0, 1)
    
    df_filtered[keep_otus] = otu_data
    
    print("Preprocessing complete. Data shape:", df_filtered.shape)
    return df_filtered, keep_otus, target_cols

###############################################################################
# 2) OPTIMIZED MANTEL TEST & GRAPH CONSTRUCTION
###############################################################################

def mantel_test_vectorized(dist_matrix1: np.ndarray, dist_matrix2: np.ndarray, 
                          permutations: int = 199) -> Tuple[float, float]:
    """Vectorized implementation of Mantel test with reduced permutations"""
    r_obs, _ = pearsonr(dist_matrix1, dist_matrix2)
    
    n = int(np.ceil((1 + np.sqrt(1 + 8 * len(dist_matrix1))) / 2))
    dist_mat2_full = squareform(dist_matrix2)
    
    # Generate all permutations at once
    perms = np.array([np.random.permutation(n) for _ in range(permutations)])
    
    # Vectorized permutation tests
    r_perms = np.array([
        pearsonr(dist_matrix1, squareform(dist_mat2_full[p][:, p]))[0]
        for p in perms
    ])
    
    p_value = (np.sum(r_perms >= r_obs) + 1) / (permutations + 1)
    return r_obs, p_value

def process_edge_pair(args: Tuple) -> Tuple[List, List]:
    """Helper function for parallel edge processing"""
    i, j, dist_vectors, features, alpha = args
    r_val, p_val = mantel_test_vectorized(dist_vectors[features[i]], 
                                         dist_vectors[features[j]])
    if p_val < alpha:
        return [(i, j), (j, i)], [r_val, r_val]
    return [], []

def build_graph_edges(df: pd.DataFrame, taxa_cols: List[str], 
                     target_cols: List[str], alpha: float = 0.05) -> Tuple:
    """Parallel implementation of graph edge construction"""
    features = taxa_cols + target_cols
    
    # Precompute distance vectors
    dist_vectors = {}
    for feat in features:
        vals = df[feat].to_numpy(dtype='float32').reshape(-1, 1)
        metric = "braycurtis" if feat in taxa_cols else "euclidean"
        dist_vectors[feat] = pdist(vals, metric=metric)
    
    # Prepare arguments for parallel processing
    edge_pairs = [(i, j, dist_vectors, features, alpha) 
                 for i in range(len(features)) 
                 for j in range(i+1, len(features))]
    
    # Process edges in parallel
    edges = []
    weights = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_edge_pair, edge_pairs))
    
    for e, w in results:
        edges.extend(e)
        weights.extend(w)
    
    if not edges:
        raise ValueError("No significant edges found. Adjust thresholds or check data.")
    
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    print(f"Graph: {len(features)} nodes, {edge_index.shape[1]} edges")
    return features, edge_index, edge_weight

###############################################################################
# 3) OPTIMIZED DATASET CREATION
###############################################################################

def create_dataset(df: pd.DataFrame, feature_list: List[str], edge_index: torch.Tensor,
                  edge_weight: torch.Tensor, target_cols: List[str], 
                  save_individual: bool = False) -> List[Data]:
    """Memory-efficient dataset creation"""
    dataset = []
    target_indices = [feature_list.index(tc) for tc in target_cols]
    
    # Pre-allocate tensors
    x_tensor = torch.zeros((len(feature_list), 1), dtype=torch.float32)
    y_tensor = torch.zeros(len(target_cols), dtype=torch.float32)
    
    for idx, row in df.iterrows():
        x_vals = row[feature_list].to_numpy(dtype='float32')
        for ti in target_indices:
            x_vals[ti] = 0.0
        
        x_tensor.copy_(torch.from_numpy(x_vals.reshape(-1, 1)))
        y_tensor.copy_(torch.from_numpy(row[target_cols].to_numpy(dtype='float32')))
        
        data = Data(
            x=x_tensor.clone(),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y_tensor.clone()
        )
        dataset.append(data)
    
    return dataset

###############################################################################
# 4) OPTIMIZED GCN MODEL
###############################################################################

class GCNPredictor(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, 
                 target_indices: List[int], dropout: float = 0.3):
        super().__init__()
        self.target_indices = target_indices
        
        # Optimized layer initialization
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_ace = nn.Linear(hidden_dim, 1)
        self.fc_h2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # Forward pass with batch normalization
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc_hidden(x))
        
        ace_idx, h2_idx = self.target_indices
        ace_pred = self.fc_ace(x[ace_idx])
        h2_pred = self.fc_h2(x[h2_idx])
        
        return torch.cat([ace_pred, h2_pred], dim=0)

###############################################################################
# 5) OPTIMIZED TRAINING WITH EARLY STOPPING
###############################################################################

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def train_model(model: nn.Module, train_data: List[Data], test_data: List[Data],
                epochs: int = 100, lr: float = 0.005, batch_size: int = 32,
                patience: int = 10) -> Tuple[nn.Module, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training on {device}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        avg_loss = total_loss / len(train_data)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train MSE: {avg_loss:.4f}")
        
        if early_stopping(avg_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final evaluation
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
    
    print(f"Final Test MSE: {mse_val:.4f} | R²: {r2_val:.4f}")
    return model, mse_val, r2_val

###############################################################################
# 6) OPTIMIZED MAIN EXECUTION
###############################################################################

def main():
    start_time = time.time()
    csv_path = "Data/New_data.csv"
    
    # 1. Data Preprocessing
    df_filtered, taxa_cols, target_cols = preprocess_data(
        csv_path,
        abundance_threshold=0.001,
        prevalence_threshold=0.01,
        otu_prefix="d__"
    )
    
    # 2. Graph Construction
    feat_list, edge_index, edge_weight = build_graph_edges(
        df_filtered, taxa_cols, target_cols, alpha=0.05
    )
    
    # 3. Dataset Creation
    dataset = create_dataset(
        df_filtered,
        feat_list,
        edge_index,
        edge_weight,
        target_cols,
        save_individual=False
    )
    
    # 4. Train/Test Split
    np.random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # 5. Model Training
    target_indices = [feat_list.index(tc) for tc in target_cols]
    model = GCNPredictor(
        num_nodes=len(feat_list),
        input_dim=1,
        hidden_dim=64,
        target_indices=target_indices,
        dropout=0.3
    )
    
    trained_model, mse_val, r2_val = train_model(
        model,
        train_data,
        test_data,
        epochs=120,
        lr=0.005,
        batch_size=32,
        patience=10
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.2f} minutes")
    print(f"Final Test MSE: {mse_val:.4f}")
    print(f"Final R²: {r2_val:.4f}")

if __name__ == "__main__":
    main() 