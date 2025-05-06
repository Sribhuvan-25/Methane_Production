# dynamic_gnn_pipeline.py
# ==========================================================
# 0) Imports & global setup
# ----------------------------------------------------------
import os, itertools, collections, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.explain import GNNExplainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================
# 1) Data loading  (unchanged from your script)
# ----------------------------------------------------------
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)
df = df[~df.isin(['x']).any(axis=1)]

otu_cols  = [c for c in df.columns if "d__" in c]
meta_cols = [c for c in df.columns if c not in otu_cols]

def extract_family(colname):
    for part in colname.split(';'):
        if part.strip().startswith('f__'):
            return part.strip()[3:] or "UnclassifiedFamily"
    return "UnclassifiedFamily"

col_to_family = {c: extract_family(c) for c in otu_cols}
family_to_cols = collections.defaultdict(list)
for c, fam in col_to_family.items(): family_to_cols[fam].append(c)

df_fam = pd.DataFrame({fam: df[cols].sum(axis=1) for fam,cols in family_to_cols.items()},
                      index=df.index)

param_df   = df[meta_cols].copy()
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)

# --- simple prevalence / abundance filter ---
presence = (df_fam_rel>0).sum(0) / df_fam_rel.shape[0]
abund    = df_fam_rel.mean(0)
keep_fams = presence[presence>=.05].index.intersection(abund[abund>=.01].index)
df_fam_rel = df_fam_rel[keep_fams]

target_cols = ['ACE-km','H2-km']
feature_matrix = df_fam_rel.apply(lambda x: np.sqrt(np.sqrt(x))).values.astype(np.float32)
targets        = param_df[target_cols].values.astype(np.float32)
node_names     = list(df_fam_rel.columns)
num_nodes      = len(node_names)
num_samples    = feature_matrix.shape[0]

print(f"Dataset: {num_samples} samples  |  {num_nodes} taxa retained")

# ==========================================================
# 2) Utility – Mantel skeleton (used by static & adaptive)
# ----------------------------------------------------------
def compute_dm(vec, metric='braycurtis'):
    return np.nan_to_num(squareform(pdist(vec[:,None], metric=metric)))

def mantel(d1,d2,perm=999):
    idx = np.triu_indices(d1.shape[0],1)
    v1,v2 = d1[idx], d2[idx]
    if v1.std()==0 or v2.std()==0: return 1.0
    r_obs = pearsonr(v1,v2)[0]
    cnt   = sum(abs(pearsonr(v1,np.random.permutation(v2))[0])>=abs(r_obs)
                for _ in range(perm))
    return (cnt+1)/(perm+1)

print("Building Mantel skeleton ...")
dist_mats = {i: compute_dm(feature_matrix[:,i]) for i in range(num_nodes)}
ei,ej = [],[]
for i,j in itertools.combinations(range(num_nodes),2):
    if mantel(dist_mats[i],dist_mats[j]) < .05:
        ei += [i,j]; ej += [j,i]
static_edge_index = torch.tensor([ei,ej], dtype=torch.long)
print("Skeleton edges:", static_edge_index.shape[1]//2)

# ==========================================================
# 3) Instance‑wise Dynamic Graph dataset
# ----------------------------------------------------------
def bray_curtis(a_i,a_j):
    s = a_i + a_j
    return 0. if s==0 else 1 - abs(a_i-a_j)/s

def build_dynamic_graph(sample_vec, thresh=.5):
    rows,cols,wts = [],[],[]
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            w = bray_curtis(sample_vec[i], sample_vec[j])
            if w < thresh: continue
            rows += [i,j]; cols += [j,i]; wts += [w,w]
    if not rows:          # fallback to fully‑connected if all below threshold
        for i in range(num_nodes):
            for j in range(i+1,num_nodes):
                rows += [i,j]; cols += [j,i]; wts += [.01,.01]
    return torch.tensor([rows,cols]), torch.tensor(wts)

class DynamicGraphDataset(InMemoryDataset):
    def __init__(self, feats, labels):
        super().__init__('')
        data_list=[]
        for f,lab in zip(feats,labels):
            e_i,e_w = build_dynamic_graph(f,thresh=.5)
            x = torch.tensor(f).unsqueeze(1)
            y = torch.tensor(lab)
            data_list.append(Data(x=x, edge_index=e_i, edge_attr=e_w, y=y))
        self.data, self.slices = self.collate(data_list)

# ==========================================================
# 4) Adaptive‑edge convolution & model
# ----------------------------------------------------------
from torch_geometric.nn import MessagePassing

class AdaptiveEdgeConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(in_ch, out_ch)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*out_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch,1))
    def forward(self,x,edge_index):
        x = self.lin_node(x)
        i,j = edge_index
        edge_feat = torch.cat([x[i],x[j]], dim=1)
        edge_w    = torch.sigmoid(self.edge_mlp(edge_feat)).squeeze()
        return self.propagate(edge_index, x=x, edge_w=edge_w)
    def message(self,x_j,edge_w): return edge_w.unsqueeze(-1)*x_j

class AdaptiveEdgeGNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = AdaptiveEdgeConv(1, hidden)
        self.conv2 = AdaptiveEdgeConv(hidden, hidden)
        self.head  = nn.Linear(hidden,1)
    def forward(self,data):
        x,edge_index,batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x,edge_index))
        x = global_mean_pool(x,batch)
        return self.head(x)

# ==========================================================
# 5) Simple GCN / GAT baseline (unchanged)
# ----------------------------------------------------------
class StaticGNN(nn.Module):
    def __init__(self, hidden=64, kind='GAT'):
        super().__init__()
        if kind=='GAT':
            self.conv1 = GATConv(1, hidden, heads=4, concat=False)
            self.conv2 = GATConv(hidden, hidden, heads=4, concat=False)
        else:
            self.conv1 = GCNConv(1, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden,1)
    def forward(self,data):
        x,edge_index,batch = data.x,data.edge_index,data.batch
        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x,edge_index))
        x = global_mean_pool(x,batch)
        return self.head(x)

# ==========================================================
# 6) Train / evaluate helpers
# ----------------------------------------------------------
def kfold_indices(n_samples: int, n_splits: int = 5, seed: int = 42):
    """
    Return list of (train_idx, val_idx) for K‑fold CV.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)  # <-- fixed
    return list(kf.split(np.arange(n_samples)))

# def run_experiment(approach='static', hidden=128, epochs=200, target_idx=0):
#     #  target_idx: 0=ACE‑km, 1=H2‑km
#     if approach=='instance_dynamic':
#         full_ds = DynamicGraphDataset(feature_matrix, targets)
#     else:
#         # build static graphs once
#         data_list=[]
#         for feats,lab in zip(feature_matrix,targets):
#             x = torch.tensor(feats).unsqueeze(1)
#             y = torch.tensor(lab)
#             data_list.append(Data(x=x, edge_index=static_edge_index, y=y))
#         full_ds, _ = InMemoryDataset('').collate(data_list)
#     splits = kfold_indices(num_samples)
#     all_preds, all_trues = [],[]
#     for fold,(tr,va) in enumerate(splits,1):
#         if approach=='instance_dynamic':
#             tr_ds = full_ds[tr]; va_ds = full_ds[va]
#         else:                   # slicing works for InMemoryDataset
#             tr_ds = full_ds.index_select(tr); va_ds = full_ds.index_select(va)
#         tr_ld = DataLoader(tr_ds,batch_size=16,shuffle=True)
#         va_ld = DataLoader(va_ds,batch_size=16)

#         # pick model
#         if approach=='static':
#             model = StaticGNN(hidden, kind='GAT').to(device)
#         elif approach=='adaptive_edge':
#             model = AdaptiveEdgeGNN(hidden).to(device)
#         elif approach=='instance_dynamic':
#             model = StaticGNN(hidden, kind='GAT').to(device)   # any conv that uses edge weights
#         opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#         for ep in range(1,epochs+1):
#             model.train(); tot=0
#             for batch in tr_ld:
#                 batch=batch.to(device); opt.zero_grad()
#                 out = model(batch).view(-1)
#                 tgt = batch.y[:,target_idx].to(device)
#                 loss = F.mse_loss(out, tgt)
#                 loss.backward(); opt.step()
#                 tot += loss.item()*batch.num_graphs
#             if ep%50==0:
#                 print(f"{approach} | Fold {fold} | Epoch {ep} | MSE {tot/len(tr_ld.dataset):.4f}")

#         # validation
#         model.eval(); preds=[]; trues=[]
#         with torch.no_grad():
#             for batch in va_ld:
#                 batch=batch.to(device)
#                 p = model(batch).view(-1).cpu()
#                 t = batch.y[:,target_idx].cpu()
#                 preds.append(p); trues.append(t)
#         all_preds.append(torch.cat(preds)); all_trues.append(torch.cat(trues))
#     all_preds = torch.cat(all_preds); all_trues = torch.cat(all_trues)
#     mse = F.mse_loss(all_preds, all_trues).item()
#     r2  = r2_score(all_trues.numpy(), all_preds.numpy())
#     print(f"\n==> {approach} result | MSE {mse:.4f} | R² {r2:.3f}\n")

#     # quick GNNExplainer demo on first validation sample of last fold
#     explainer = GNNExplainer(model, epochs=100, return_type='regression')
#     test_graph = va_ds[0].to(device)
#     feat_mask, edge_mask = explainer.explain_graph(test_graph.x,
#                                                    test_graph.edge_index)
#     important = (feat_mask.sigmoid()>0.5).nonzero(as_tuple=True)[0]
#     print("Top‑influence taxa indices:", important.tolist()[:10])
#     return mse,r2

def run_experiment(approach='static', hidden=128, epochs=200, target_idx=0):
    # 0) choose/build dataset or list of graphs --------------------------------
    if approach == 'instance_dynamic':
        full_ds = DynamicGraphDataset(feature_matrix, targets)  # real Dataset
    else:
        # keep a *list* of Data objects for static/adaptive runs
        static_graphs = []
        for feats, lab in zip(feature_matrix, targets):
            x = torch.tensor(feats).unsqueeze(1)
            y = torch.tensor(lab)
            static_graphs.append(Data(x=x,
                                      edge_index=static_edge_index,
                                      y=y))
        full_ds = static_graphs  # <-- just a Python list

    # 1) CV splits -------------------------------------------------------------
    splits = kfold_indices(num_samples)

    # 2) loop over folds -------------------------------------------------------
    all_preds, all_trues = [], []
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):

        # ---- slice the data --------------------------------------------------
        if approach == 'instance_dynamic':              # Dataset supports fancy indexing
            tr_ds = full_ds.index_select(tr_idx)
            va_ds = full_ds.index_select(va_idx)
        else:                                           # slice a plain list
            tr_ds = [full_ds[i] for i in tr_idx]
            va_ds = [full_ds[i] for i in va_idx]

        tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=16)

        # ---- pick model ------------------------------------------------------
        if approach == 'static':
            model = StaticGNN(hidden, kind='GAT').to(device)
        elif approach == 'adaptive_edge':
            model = AdaptiveEdgeGNN(hidden).to(device)
        else:  # instance_dynamic
            model = StaticGNN(hidden, kind='GAT').to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # ---------- training loop -----------------------------------------------
        for ep in range(1, epochs + 1):
            model.train(); tot = 0.0
            for batch in tr_ld:
                batch = batch.to(device)
                opt.zero_grad()
                out = model(batch).view(-1)               # [B]
                # --- handle 1‑D vs 2‑D y -------------
                if batch.y.dim() == 1:                    # shape [B*2]
                    y_batch = batch.y.view(batch.num_graphs, -1)  # → [B,2]
                else:
                    y_batch = batch.y                    # already [B,2]
                tgt = y_batch[:, target_idx].to(device)   # [B]
                loss = F.mse_loss(out, tgt)
                loss.backward(); opt.step()
                tot += loss.item() * batch.num_graphs
            if ep % 50 == 0:
                print(f"{approach} | Fold {fold} | Epoch {ep} | "
                    f"MSE {tot/len(tr_ld.dataset):.4f}")

        # ---------- validation ---------------------------------------------------
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for batch in va_ld:
                batch = batch.to(device)
                out = model(batch).view(-1).cpu()
                if batch.y.dim() == 1:
                    y_val = batch.y.view(batch.num_graphs, -1)[:, target_idx].cpu()
                else:
                    y_val = batch.y[:, target_idx].cpu()
                preds.append(out); trues.append(y_val)
        all_preds.append(torch.cat(preds)); all_trues.append(torch.cat(trues))


    # 3) metrics ---------------------------------------------------------------
    all_preds = torch.cat(all_preds); all_trues = torch.cat(all_trues)
    mse = F.mse_loss(all_preds, all_trues).item()
    r2  = r2_score(all_trues.numpy(), all_preds.numpy())
    print(f"\n==> {approach} result | MSE {mse:.4f} | R² {r2:.3f}\n")

    # 4) (Optional) quick GNNExplainer demo on one sample ----------------------
    if approach != 'instance_dynamic':  # use a static graph example
        explainer = GNNExplainer(model, epochs=100, return_type='regression')
        test_graph = va_ds[0].to(device)
        feat_mask, edge_mask = explainer.explain_graph(test_graph.x,
                                                       test_graph.edge_index)
        important = (feat_mask.sigmoid() > 0.5).nonzero(as_tuple=True)[0]
        print("Top‑influence taxa indices:", important.tolist()[:10])

    return mse, r2


print("Training for ACE‑km …")
run_experiment('static',          target_idx=0)
run_experiment('instance_dynamic',target_idx=0)
run_experiment('adaptive_edge',   target_idx=0)

print("Training for H2‑km …")
run_experiment('static',          target_idx=1)
run_experiment('instance_dynamic',target_idx=1)
run_experiment('adaptive_edge',   target_idx=1)
