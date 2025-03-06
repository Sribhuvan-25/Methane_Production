import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_squared_error, r2_score

class GCNPredictor(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, target_indices, dropout=0.3):
        super().__init__()
        self.target_indices = target_indices
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_ace = nn.Linear(hidden_dim, 1)
        self.fc_h2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.fc_hidden(x))
        ace_idx, h2_idx = self.target_indices
        ace_pred = self.fc_ace(x[ace_idx])
        h2_pred = self.fc_h2(x[h2_idx])
        return torch.cat([ace_pred, h2_pred], dim=0)

def train_model(model, train_data, test_data, epochs=100, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train MSE: {avg_loss:.4f}")
    
    # Evaluate
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    import numpy as np
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mse_val = mean_squared_error(trues, preds)
    r2_val = r2_score(trues, preds)
    print(f"Test MSE: {mse_val:.4f} | Test R^2: {r2_val:.4f}")
    return model, mse_val, r2_val

def main():
    # Load the dataset from the first file
    dataset, feat_list, target_cols = torch.load("graph_dataset.pt")
    print(f"Loaded dataset of size {len(dataset)} from graph_dataset.pt")
    
    # Shuffle and split
    import numpy as np
    np.random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    target_indices = [feat_list.index(tc) for tc in target_cols]
    model = GCNPredictor(
        num_nodes=len(feat_list),
        input_dim=1,
        hidden_dim=64,
        target_indices=target_indices,
        dropout=0.3
    )
    print("Starting training...")
    trained_model, mse_val, r2_val = train_model(
        model,
        train_data,
        test_data,
        epochs=50,
        lr=0.005
    )
    print("Training done.")

if __name__ == "__main__":
    main()
