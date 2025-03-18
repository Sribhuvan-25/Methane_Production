##############################################################################
# Complete Example: Building Hierarchical Graphs from "New_data.csv" for GCN #
##############################################################################

import pandas as pd
import numpy as np
import networkx as nx
import torch
import os
import matplotlib.pyplot as plt

from torch_geometric.utils import from_networkx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
# If you want a Mantel test, you may need to implement or use another library, e.g. 'py mantel' (not in stdlib).

###############################################################################
# 1. Read the CSV and Identify Relevant Columns
###############################################################################
# Assumptions (please adapt to your file's structure):
#  - Each row in New_data.csv corresponds to one sample or observation.
#  - There are columns for sample-level metadata (Waste, Biomass, etc.).
#  - There are columns for the target variables ACE-km and H2-km.
#  - There is a column containing the taxonomy string OR separate columns for
#    domain, phylum, class, order, family, genus. If your file has multiple OTUs
#    per sample, you might need to reshape or group the data. The script below
#    assumes one row per sample with a single "Taxonomy" column for demonstration.
#
# IMPORTANT: If your dataset is structured differently (e.g. multiple OTUs
# per row, or columns for each OTU), you must adapt the parsing logic.

# Replace "New_data.csv" with your actual file path
data_path = "../Data/New_data.csv"
df = pd.read_csv(data_path)

# Example: Let's assume columns in your CSV (you will adapt these names):
# "Sample_ID", "Waste", "Biomass", "Average_Total_ISD_Cells",
# "ACE-km", "H2-km", "Taxonomy", "Relative_Abundance"
#
# The "Taxonomy" column might have strings like:
# "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus"
#
# If your format is different, you must adapt accordingly.

# For clarity, define which columns are your features and targets:
FEATURE_COLS = ["Waste", "Biomass", "Average_Total_ISD_Cells"] # "ACE-Xi", "ACE-Ks", "H2-Xi", "H2-Ks",
FEATURE_COLS = [
    # "Average-Total-ISD-Cells"
    # One-hot encoded 'Digester_' columns
    "Digester_BD", "Digester_BF", "Digester_CB", "Digester_CP", "Digester_FD",
    "Digester_GB", "Digester_GP", "Digester_JB", "Digester_LP", "Digester_MA",
    "Digester_NB", "Digester_NS", "Digester_PC", "Digester_PO", "Digester_SF",
    "Digester_SS", "Digester_SW", "Digester_WA", "Digester_WP", "Digester_WR",
    # One-hot encoded 'Source_' columns
    "Source_I", "Source_M", "Source_P",
    # One-hot encoded 'Type_' columns
    "Type_CSTR", "Type_EFB", "Type_EGSB", "Type_Lagoon", "Type_UASB",
    # One-hot encoded 'Waste_' columns
    "Waste_BW", "Waste_Dairy", "Waste_FW", "Waste_HSI", "Waste_MPW",
    "Waste_MS", "Waste_MS+Dairy", "Waste_MS+HSI", "Waste_PP", "Waste_PR",
    "Waste_SDW",
    # One-hot encoded 'Biomass_' columns
    "Biomass_F", "Biomass_G"
]
TARGET_COLS  = ["ACE-km", "H2-km"]  # focusing on these two

# If you have a single row per sample, but multiple taxa in the same row, you must restructure.
# For example, you might have columns for many OTUs. That scenario requires pivoting or melting.
# Below, we assume each row is a sample with exactly one taxonomy and a "Relative_Abundance" measure.
# If you actually have multiple OTUs per sample in separate columns, you'd pivot them into long form.

# Identify OTU columns (columns not in features or targets)
otu_cols = [col for col in df.columns 
            if col not in FEATURE_COLS + TARGET_COLS + ["Sample_ID"]]

print(f"Found {len(otu_cols)} OTU columns")
print("Example OTU column name:", otu_cols[0])  # Let's see one example

def parse_taxonomy_string(otu_name):
    """
    Parse OTU column name containing taxonomy levels
    Format: "d__Bacteria;p__Phylum;c__Class;o__Order;f__Family;g__Genus"
    """
    tax_levels = {
        "domain": None,
        "phylum": None,
        "class": None,
        "order": None,
        "family": None,
        "genus": None
    }
    
    parts = otu_name.split(';')
    for part in parts:
        if part.startswith('d__'):
            tax_levels["domain"] = part[3:].strip()
        elif part.startswith('p__'):
            tax_levels["phylum"] = part[3:].strip()
        elif part.startswith('c__'):
            tax_levels["class"] = part[3:].strip()
        elif part.startswith('o__'):
            tax_levels["order"] = part[3:].strip()
        elif part.startswith('f__'):
            tax_levels["family"] = part[3:].strip()
        elif part.startswith('g__'):
            tax_levels["genus"] = part[3:].strip()
    
    return tax_levels

###############################################################################
# 2. Organize Data into a More Convenient Structure
###############################################################################
# We want to build one graph per sample ID. So we group the CSV by sample.
# Then within each sample, we might have multiple taxonomy entries if your data
# has multiple rows per sample (i.e. multiple OTUs per sample).
# We'll create a dictionary: sample_data[sample_id] -> {
#   'features': { "Waste": val, "Biomass": val, ... },
#   'targets': { "ACE-km": val, "H2-km": val },
#   'taxa': [ { 'domain': ..., 'phylum': ..., 'class': ..., 'order': ...,
#               'family': ..., 'genus': ..., 'abundance': float }, ... ]
# }

sample_data = {}

# If your CSV has a column "Sample_ID" to identify samples:
if "Sample_ID" not in df.columns:
    # If you don't have a sample ID column, we'll artificially create one from index
    df["Sample_ID"] = df.index.astype(str)

# First, remove rows containing 'x'
df = df[~df.isin(['x']).any(axis=1)]

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Remove outliers where ACE-km > 40
df = df[df['ACE-km'] <= 40]

# Group by sample
for sample_id, row in df.iterrows():
    features_dict = {col: float(row[col]) for col in FEATURE_COLS}
    targets_dict = {col: float(row[col]) for col in TARGET_COLS}
    
    # Process OTUs
    taxa_list = []
    for otu in otu_cols:
        abundance = float(row[otu])
        if abundance > 0:  # Only include OTUs present in this sample
            tax_info = parse_taxonomy_string(otu)
            tax_info["abundance"] = abundance
            taxa_list.append(tax_info)
    
    sample_data[str(sample_id)] = {
        "features": features_dict,
        "targets": targets_dict,
        "taxa": taxa_list
    }


###############################################################################
# 3. (Optional) Compute Microbe–Microbe Similarity (e.g., Bray–Curtis) 
#    or Correlations for Edges Across Samples
###############################################################################
# The paper uses the Mantel test to find associations among families (or genera).
# We'll show a simpler approach: compute the correlation among families based on
# their abundance across all samples. Then we can decide if we add an edge for
# pairs with p < 0.05. If you want Bray–Curtis + Mantel test, you'll need a bit
# more advanced approach. This is just a demonstration.

# 3.1 Collect all families across samples
all_families = set()
for s_id, s_info in sample_data.items():
    for tax_info in s_info["taxa"]:
        fam = tax_info.get("family", None)
        if fam:
            all_families.add(fam)
all_families = sorted(list(all_families))

# 3.2 Build a family-by-sample abundance matrix (rows=families, cols=samples)
sample_ids_unique = sorted(sample_data.keys())
fam_array = np.zeros((len(all_families), len(sample_ids_unique)), dtype=float)

fam_index = {fam: i for i, fam in enumerate(all_families)}
sample_index = {sid: i for i, sid in enumerate(sample_ids_unique)}

for sid, s_info in sample_data.items():
    for tax_info in s_info["taxa"]:
        fam = tax_info.get("family", None)
        if fam in fam_index:
            fam_array[fam_index[fam], sample_index[sid]] += tax_info["abundance"]

# 3.3 Compute pairwise correlations between families across samples
#     We'll store significantly correlated family pairs in a set or list.
significant_family_pairs = []
alpha = 0.05  # significance threshold
for i in range(len(all_families)):
    for j in range(i+1, len(all_families)):
        fam1 = all_families[i]
        fam2 = all_families[j]
        vec1 = fam_array[i, :]
        vec2 = fam_array[j, :]
        # If all zeros, skip
        if np.allclose(vec1, 0) or np.allclose(vec2, 0):
            continue
        r, pval = pearsonr(vec1, vec2)
        if pval < alpha:
            significant_family_pairs.append((fam1, fam2))

# (If you want Bray–Curtis distance, you can do something like:)
# bc_dist = pdist(fam_array, metric='braycurtis')
# bc_mat = squareform(bc_dist)
# Then do a Mantel test or threshold. Not shown here.

###############################################################################
# 4. Define Graph Saving Functions
###############################################################################

def compute_mantel_test(matrix_a, matrix_b, n_permutations=999):
    """
    Perform Mantel test between two distance matrices
    Returns correlation coefficient and p-value
    """
    orig_corr = pearsonr(matrix_a.flatten(), matrix_b.flatten())[0]
    
    # Permutation test
    n_better = 0
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(len(matrix_b))
        perm_matrix = matrix_b[perm_idx][:, perm_idx]
        perm_corr = pearsonr(matrix_a.flatten(), perm_matrix.flatten())[0]
        if abs(perm_corr) >= abs(orig_corr):
            n_better += 1
    
    pvalue = (n_better + 1) / (n_permutations + 1)
    return orig_corr, pvalue

def compute_feature_microbe_correlations(df, feature_cols, otu_cols):
    """
    Compute correlations between features and microbes with FDR correction
    Returns dict of significant feature-microbe pairs
    """
    correlations = []
    for feature in feature_cols:
        for otu in otu_cols:
            rho, p = spearmanr(df[feature], df[otu])
            correlations.append((feature, otu, rho, p))
    
    # FDR correction
    _, p_adjusted, _, _ = multipletests(
        [c[3] for c in correlations], 
        method='fdr_bh'
    )
    
    significant_pairs = {}
    for (feature, otu, rho, _), p_adj in zip(correlations, p_adjusted):
        if p_adj < 0.05:
            if feature not in significant_pairs:
                significant_pairs[feature] = []
            significant_pairs[feature].append((otu, rho))
    
    return significant_pairs

def save_detailed_graph_data(G, sample_id, output_dir='graph_data'):
    """
    Enhanced graph data saving with additional statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f'graph_{sample_id}_info.txt'), 'w') as f:
        # Basic graph information
        f.write(f"Graph Information for Sample {sample_id}\n")
        f.write("=" * 50 + "\n\n")
        
        # Node statistics by type
        f.write("Node Statistics:\n")
        f.write("-" * 20 + "\n")
        node_types = set(nx.get_node_attributes(G, 'node_type').values())
        for ntype in node_types:
            count = len([n for n, d in G.nodes(data=True) if d['node_type'] == ntype])
            f.write(f"{ntype}: {count} nodes\n")
        
        # Edge statistics by type
        f.write("\nEdge Statistics:\n")
        f.write("-" * 20 + "\n")
        edge_types = set(nx.get_edge_attributes(G, 'edge_type').values())
        for etype in edge_types:
            count = len([e for e in G.edges(data=True) if e[2]['edge_type'] == etype])
            f.write(f"{etype}: {count} edges\n")
        
        # Target values
        f.write("\nTarget Values:\n")
        f.write("-" * 20 + "\n")
        f.write(f"ACE-km: {G.graph['target'][0]:.4f}\n")
        f.write(f"H2-km: {G.graph['target'][1]:.4f}\n")
        
        # Network metrics
        f.write("\nNetwork Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average clustering coefficient: {nx.average_clustering(G):.4f}\n")
        f.write(f"Graph density: {nx.density(G):.4f}\n")
        
        # Detailed node information
        f.write("\nDetailed Node Information:\n")
        f.write("-" * 20 + "\n")
        for node, data in G.nodes(data=True):
            f.write(f"\nNode: {node}\n")
            f.write(f"Type: {data['node_type']}\n")
            f.write(f"Feature value: {data['features'][0]:.6f}\n")
            f.write(f"Degree: {G.degree(node)}\n")

def enhance_graph_visualization(G, sample_id, output_dir='graph_visualizations'):
    """
    Enhanced graph visualization with better layout and information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 15))
    
    # Use hierarchical layout for better taxonomy visualization
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Node colors by type with better contrast
    node_colors = {
        'domain': '#1f77b4',    # Blue
        'phylum': '#ff7f0e',    # Orange
        'class': '#2ca02c',     # Green
        'order': '#d62728',     # Red
        'family': '#9467bd',    # Purple
        'genus': '#8c564b',     # Brown
        'feature': '#e377c2'    # Pink
    }
    
    # Edge colors by type
    edge_colors = {
        'taxonomy': '#000000',          # Black
        'fam_correlation': '#ff0000',   # Red
        'feature_assoc': '#00ff00'      # Green
    }
    
    # Draw nodes with size based on feature value
    node_sizes = []
    node_colors_list = []
    for node in G.nodes():
        node_type = G.nodes[node]['node_type']
        feature_val = G.nodes[node]['features'][0]
        
        # Scale node size by feature value
        size = np.clip(feature_val * 1000, 100, 2000)
        node_sizes.append(size)
        
        # Get color from mapping
        node_colors_list.append(node_colors.get(node_type, '#7f7f7f'))
    
    # Draw edges with different colors and styles
    for edge_type in edge_colors:
        edge_list = [(u, v) for (u, v, d) in G.edges(data=True) 
                    if d['edge_type'] == edge_type]
        if edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=edge_list,
                                 edge_color=edge_colors[edge_type],
                                 alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=node_colors_list, alpha=0.8)
    
    # Add labels with smaller font for better readability
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color, label=node_type,
                                 markersize=10)
                      for node_type, color in node_colors.items()]
    legend_elements.extend([plt.Line2D([0], [0], color=color,
                                     label=edge_type)
                          for edge_type, color in edge_colors.items()])
    
    plt.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1, 0.5))
    
    plt.title(f'Hierarchical Graph Structure for Sample {sample_id}')
    plt.axis('off')
    plt.tight_layout()
    
    # Save with high resolution
    plt.savefig(os.path.join(output_dir, f'graph_{sample_id}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

###############################################################################
def save_graph_data(G, sample_id, output_dir='graph_data'):
    """
    Save graph data in a text format for inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f'graph_{sample_id}_info.txt'), 'w') as f:
        # Write basic graph information
        f.write(f"Graph Information for Sample {sample_id}\n")
        f.write("=" * 50 + "\n\n")
        
        # Node information
        f.write("Nodes:\n")
        f.write("-" * 20 + "\n")
        for node in G.nodes(data=True):
            f.write(f"Node: {node[0]}\n")
            f.write(f"Type: {node[1]['node_type']}\n")
            f.write(f"Features: {node[1]['features']}\n")
            f.write("\n")
        
        # Edge information
        f.write("\nEdges:\n")
        f.write("-" * 20 + "\n")
        for edge in G.edges(data=True):
            f.write(f"Edge: {edge[0]} -> {edge[1]}\n")
            f.write(f"Type: {edge[2].get('edge_type', 'default')}\n")
            f.write("\n")
        
        # Graph-level information
        f.write("\nGraph-level Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of nodes: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n")
        f.write(f"Target values: {G.graph['target']}\n")

###############################################################################
# 5. Build One Graph per Sample in NetworkX
###############################################################################
# We'll connect:
#   (A) Hierarchical taxonomy: domain -> phylum -> class -> order -> family -> genus
#   (B) Microbe–microbe edges for significantly correlated families
#   (C) Feature nodes for each sample, connected if correlation with microbe is significant 
#       (here, for simplicity, we skip feature->microbe correlation edges or do a naive approach)
#
# Finally, each graph gets target = [ACE-km, H2-km].
###############################################################################

data_list = []  # will hold torch_geometric Data objects

for sid in sample_ids_unique:
    s_info = sample_data[sid]
    G = nx.Graph()
    
    # 4.1 Add sample-level feature nodes
    for feat_name, val in s_info["features"].items():
        # Make a node for each feature
        G.add_node(feat_name, features=[val], node_type="feature")
    
    # 4.2 Add taxonomy nodes and connect them hierarchically
    #     We'll keep a dictionary to track if a node is already added
    #     for domain, phylum, etc.
    
    # A small helper to safely add a node to G if it doesn't exist
    # and set/update its feature. We store the sum of abundances if added multiple times.
    def add_tax_node(node_id, node_level, abundance):
        if node_id not in G:
            G.add_node(node_id, features=[abundance], node_type=node_level)
        else:
            # if it already exists, add abundance (or do something appropriate)
            current_val = G.nodes[node_id]["features"][0]
            G.nodes[node_id]["features"][0] = current_val + abundance
    
    # For each taxon in this sample:
    for tax_info in s_info["taxa"]:
        dom = tax_info.get("domain", None)
        phy = tax_info.get("phylum", None)
        cla = tax_info.get("class", None)
        ord_ = tax_info.get("order", None)
        fam = tax_info.get("family", None)
        gen = tax_info.get("genus", None)
        abund = tax_info["abundance"]
        
        # Add or update nodes in hierarchy
        # Domain
        if dom:
            add_tax_node(dom, "domain", abund)
        if phy:
            add_tax_node(phy, "phylum", abund)
            if dom and dom in G:
                G.add_edge(dom, phy, edge_type="taxonomy")
        if cla:
            add_tax_node(cla, "class", abund)
            if phy and phy in G:
                G.add_edge(phy, cla, edge_type="taxonomy")
        if ord_:
            add_tax_node(ord_, "order", abund)
            if cla and cla in G:
                G.add_edge(cla, ord_, edge_type="taxonomy")
        if fam:
            add_tax_node(fam, "family", abund)
            if ord_ and ord_ in G:
                G.add_edge(ord_, fam, edge_type="taxonomy")
        if gen:
            add_tax_node(gen, "genus", abund)
            if fam and fam in G:
                G.add_edge(fam, gen, edge_type="taxonomy")
    
    # 4.3 Add microbe–microbe edges for significantly correlated families
    #     If both families exist in the sample, connect them.
    #     We'll skip weighting edges by correlation for simplicity.
    families_in_sample = [n for n, d in G.nodes(data=True) if d["node_type"] == "family"]
    families_in_sample_set = set(families_in_sample)
    for (fam1, fam2) in significant_family_pairs:
        if fam1 in families_in_sample_set and fam2 in families_in_sample_set:
            G.add_edge(fam1, fam2, edge_type="fam_correlation")
    
    # 4.4 (Optional) Add microbe–feature edges if correlated in this sample
    #     For brevity, let's skip or do a naive approach. You could compute
    #     correlations on the fly: feature vs. each family's abundance across samples
    #     or just link them if p < 0.05. The code below is a placeholder.
    
    # e.g., we link if the sample's feature value is > some threshold and abundance is > 0
    # for feat_name, val in s_info["features"].items():
    #     for fam_node in families_in_sample:
    #         # check if that family node has some abundance in this sample
    #         if G.nodes[fam_node]["features"][0] > 0 and val > 0:
    #             G.add_edge(feat_name, fam_node, edge_type="feature_association")
    
    # 4.5 Attach the target to the graph
    # We assume ACE-km and H2-km are in s_info["targets"]
    ace_km = s_info["targets"].get("ACE-km", 0.0)
    h2_km  = s_info["targets"].get("H2-km", 0.0)
    G.graph["target"] = [ace_km, h2_km]
    
    # Before converting to torch_geometric Data, ensure all edges have edge_type:
    for u, v in G.edges():
        if 'edge_type' not in G[u][v]:
            G[u][v]['edge_type'] = "default"  # Add a default edge type if missing
    
    # Save visualization and data before converting to PyG Data
    enhance_graph_visualization(G, sid)
    save_detailed_graph_data(G, sid)
    
    # 4.6 Convert to torch_geometric Data
    data = from_networkx(G, group_node_attrs=['features'])
    
    # Manually add edge attributes after conversion
    edge_types = []
    for u, v in G.edges():
        edge_types.append(G[u][v].get('edge_type', 'default'))
    
    # Convert edge types to tensor
    data.edge_attr = torch.tensor([1 if et == 'taxonomy' else 2 if et == 'fam_correlation' else 0 
                                 for et in edge_types], dtype=torch.long)
    
    # store the target
    data.y = torch.tensor(G.graph["target"], dtype=torch.float)
    
    # optionally store a sample ID in data if you want
    data.sample_id = sid
    
    data_list.append(data)

###############################################################################
# 5. Now data_list contains a torch_geometric Data object for each sample.
#    You can combine them into a dataset or DataLoader for GCN training.
###############################################################################

print(f"Created {len(data_list)} graphs. Example of the first graph:")
print(data_list[0])
print("Node features shape:", data_list[0].x.shape)
print("Edge index shape:", data_list[0].edge_index.shape)
print("Targets:", data_list[0].y)

print("\nGraph visualizations have been saved in 'graph_visualizations' directory")
print("Graph data has been saved in 'graph_data' directory")

# Optional: Save the PyG data list for later use
torch.save(data_list, 'graph_data/pyg_data_list.pt')
print("PyG data list saved to 'graph_data/pyg_data_list.pt'")
