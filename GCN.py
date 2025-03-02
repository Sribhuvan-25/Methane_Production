import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import scipy.sparse as sp
from torch_geometric.loader import DataLoader  # Change from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool

class OTUGraphDataset:
    def __init__(self, data_path: str, threshold: float = 0.5):
        self.data_path = data_path
        self.threshold = threshold
        self.graphs = []
        self.load_and_process_data()
        
        # Add these properties after loading data
        self.num_node_features = 1  # Since we're using normalized abundances as features
        self.num_classes = len(self.metadata_cols)  # Number of metadata columns to predict

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Return the graph at the specified index."""
        return self.graphs[idx]

    def load_and_process_data(self):
        self.df = pd.read_csv(self.data_path, index_col=0)
        initial_samples = len(self.df)
        
        self.otu_cols = [col for col in self.df.columns if ';' in col]
        self.metadata_cols = [col for col in self.df.columns if ';' not in col]
        
        # Convert metadata columns to numeric and drop rows with non-numeric values
        numeric_mask = self.df[self.metadata_cols].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
        self.df = self.df[numeric_mask]
        
        # Convert metadata to numeric type after filtering
        for col in self.metadata_cols:
            self.df[col] = pd.to_numeric(self.df[col])
        
        print(f"Removed {initial_samples - len(self.df)} samples with non-numeric metadata values")
        print(f"Remaining samples: {len(self.df)}")
        
        for idx in range(len(self.df)):
            graph = self.create_sample_graph(idx)
            self.graphs.append(graph)

    def create_sample_graph(self, sample_idx: int) -> Data:
        otu_data = self.df.iloc[sample_idx][self.otu_cols].values
        node_features = self.create_node_features(otu_data)
        edge_index, edge_attr = self.create_edges(otu_data)
        
        # Ensure metadata is float type and correct shape
        metadata = self.df.iloc[sample_idx][self.metadata_cols].values.astype(float)
        
        # Debug prints
        print(f"\nSample {sample_idx}:")
        print(f"Node features shape: {node_features.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Metadata shape: {metadata.shape}")
        
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.FloatTensor(metadata)
        )
        
        return data

    def create_node_features(self, otu_abundances: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        normalized_abundances = scaler.fit_transform(otu_abundances.reshape(-1, 1))
        node_features = normalized_abundances
        return node_features

    def create_edges(self, otu_abundances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create a fully connected graph since we can't calculate correlations
        # for a single sample
        num_nodes = len(otu_abundances)
        edges_source = []
        edges_target = []
        edge_weights = []
        
        # Create edges between all pairs of nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    edges_source.append(i)
                    edges_target.append(j)
                    # Use the product of normalized abundances as edge weight
                    weight = otu_abundances[i] * otu_abundances[j]
                    edge_weights.append(weight)
        
        edge_index = np.array([edges_source, edges_target])
        edge_attr = np.array(edge_weights)
        
        return edge_index, edge_attr

class GCN(nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 64):
        super(GCN, self).__init__()
        self.num_classes = num_classes  # Store for later use
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (during testing with single graphs), create a dummy batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Final prediction
        out = self.classifier(x)
        return out

def train_gcn(model: GCN, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
              criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        # Debug prints
        print(f"\nBatch {batch_idx}:")
        print(f"x shape: {data.x.shape}")
        print(f"y shape: {data.y.shape}")
        print(f"batch shape: {data.batch.shape}")
        print(f"out shape: {out.shape}")
        
        # Reshape target if needed
        if len(data.y.shape) == 1:
            data.y = data.y.view(-1, model.num_classes)
        
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_gcn(model: GCN, test_loader: DataLoader, criterion: nn.Module,
                device: torch.device) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = OTUGraphDataset('Data/New_Data.csv', threshold=0.5)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Print sample data structure
    sample_data = dataset[0]
    print("\nSample data structure:")
    print(f"x shape: {sample_data.x.shape}")
    print(f"y shape: {sample_data.y.shape}")
    print(f"edge_index shape: {sample_data.edge_index.shape}")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Use PyTorch Geometric's DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = GCN(
        num_node_features=dataset.num_node_features,
        num_classes=dataset.num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        train_loss = train_gcn(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate_gcn(model, test_loader, criterion, device)
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    main()