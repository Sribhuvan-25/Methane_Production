#%%
import pandas as pd
import collections
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
#%%
data_path = "../Data/New_data.csv"

df = pd.read_csv(data_path)
df = df[~df.isin(['x']).any(axis=1)]
print(df.columns)
otu_cols = [col for col in df.columns if "d__" in col]
meta_cols = [col for col in df.columns if col not in otu_cols]

def extract_family_from_colname(colname):
    parts = colname.split(';')
    for part in parts:
        part = part.strip()
        if part.startswith('f__'):
            fam = part[3:]
            return fam if fam else "UnclassifiedFamily"
    return "UnclassifiedFamily"

col_to_family = {}
for c in otu_cols:
    family = extract_family_from_colname(c)
    col_to_family[c] = family

family_to_cols = collections.defaultdict(list)
for c in otu_cols:
    fam = col_to_family[c]
    family_to_cols[fam].append(c)

df_fam = pd.DataFrame(index=df.index)
for fam, col_list in family_to_cols.items():
    df_fam[fam] = df[col_list].sum(axis=1)

param_df = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

df_fam_rel.to_csv("family_abundance.csv", index=False)
param_df.to_csv("parameters.csv", index=False)

#%%
# Filter microbial families by prevalance and abundance
df_fam_rel = pd.read_csv("family_abundance.csv")
param_df = pd.read_csv("parameters.csv")

presence_count = (df_fam_rel > 0).sum(axis=0)
prevalence = presence_count / df_fam_rel.shape[0]
prev_threshold = 0.10
high_prev_families = prevalence[prevalence >= prev_threshold].index

mean_abund = df_fam_rel.mean(axis=0)
abund_threshold = 0.001
high_abund_families = mean_abund[mean_abund >= abund_threshold].index

selected_families = high_prev_families.intersection(high_abund_families)
df_fam_rel_filtered = df_fam_rel[selected_families].copy()
print(f"Selected {len(selected_families)} families after filtering.")

# %%
# Construct static graph nodes (Microbial families + Parameter nodes)

target_cols = ['ACE-km', 'H2-km']
param_inputs = param_df.drop(columns=target_cols, errors='ignore')

df_microbe_sqrt = df_fam_rel_filtered.apply(lambda x: np.sqrt(np.sqrt(x)))

all_features_df = pd.concat([df_microbe_sqrt, param_inputs], axis=1)
print("Combined feature matrix shape (samples x nodes):", all_features_df.shape)

# %%
# Define edges based on pairwise correlations

corr_matrix = all_features_df.corr()
corr_threshold = 0.4
edge_index_list = [[], []]
num_nodes = corr_matrix.shape[0]
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if abs(corr_matrix.iat[i, j]) >= corr_threshold:
            # Add an undirected edge (i <-> j) as two directed edges i->j and j->i
            edge_index_list[0].extend([i, j])
            edge_index_list[1].extend([j, i])
edge_index = torch.tensor(edge_index_list, dtype=torch.long)
print("Total edges (undirected) defined by threshold:", edge_index.shape[1]//2)

# %%
# Create graph Data objects for each sample

param_df = pd.read_csv("parameters.csv")
param_df.columns = param_df.columns.str.strip()
all_features_df = pd.concat([df_microbe_sqrt, param_inputs], axis=1)
all_features_df = all_features_df.apply(pd.to_numeric, errors='coerce')
print("Combined feature matrix shape (samples x nodes):", all_features_df.shape)

node_features = all_features_df.values.astype(np.float32)
target_subset = param_df[target_cols].copy()

data_list = []
for idx in range(len(all_features_df)):
    x = torch.tensor(node_features[idx], dtype=torch.float).unsqueeze(1)
    # Now target_subset has only the 2 desired columns, so this returns a 2-element array
    y_vals = target_subset.iloc[idx].values.astype(np.float32)
    y = torch.tensor(y_vals, dtype=torch.float).unsqueeze(0)
    graph = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(graph)

print(f"Created {len(data_list)} graph data objects (one per sample).")



# %%
# Define GCN Model

class GCNRegression(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNRegression, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.lin(x)
        return out

model = GCNRegression(num_node_features=1, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


# %%
# Train GCN Model

train_idx, test_idx = train_test_split(np.arange(len(data_list)), test_size=0.2, random_state=42)
train_dataset = [data_list[i] for i in train_idx]
test_dataset  = [data_list[i] for i in test_idx]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training loop
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for batch_data in train_loader:
        optimizer.zero_grad()
        out = model(batch_data)
        loss = criterion(out, batch_data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_data.num_graphs
    avg_loss = total_loss / len(train_dataset)
    if epoch % 10 == 0:
        model.eval()
        # Compute training loss
        train_loss = 0
        for batch_data in train_loader:
            pred = model(batch_data)
            train_loss += criterion(pred, batch_data.y).item() * batch_data.num_graphs
        train_loss /= len(train_dataset)
        # Compute test loss
        test_loss = 0
        for batch_data in test_loader:
            pred = model(batch_data)
            test_loss += criterion(pred, batch_data.y).item() * batch_data.num_graphs
        test_loss /= len(test_dataset)
        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")


# %%
