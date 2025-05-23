import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from torch_geometric.data import Data
import networkx as nx
from scipy.spatial.distance import jensenshannon

class MicrobialGNNDataset:
    """Dataset class for GNN-based regression on microbial data"""
    
    def __init__(self, data_path, k_neighbors=5, mantel_threshold=0.05, use_fast_correlation=True, graph_mode='otu'):
        """
        Initialize the dataset class
        
        Args:
            data_path: Path to the CSV file with microbial data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test (only used if use_fast_correlation=False)
            use_fast_correlation: If True, use fast correlation-based graph construction.
                                If False, use slower but more rigorous Mantel test approach.
            graph_mode: 'otu' for OTU-based graphs, 'family' for family-level aggregated graphs
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.use_fast_correlation = use_fast_correlation
        self.graph_mode = graph_mode
        
        # Load and process the data
        self.df = pd.read_csv(data_path)
        
        # Filter out rows containing 'x' values if they exist
        if self.df.isin(['x']).any().any():
            self.df = self.df[~self.df.isin(['x']).any(axis=1)]
        
        # Identify feature and target columns based on graph mode
        # Use only the two specific target columns requested
        self.target_cols = ['ACE-km', 'H2-km']
        
        # Check if the target columns exist in the data
        missing_targets = [col for col in self.target_cols if col not in self.df.columns]
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
            # Use only the available target columns
            self.target_cols = [col for col in self.target_cols if col in self.df.columns]
        
        if self.graph_mode == 'otu':
            # Identify OTU feature columns (likely with taxonomic identifiers)
            # Look for columns with taxonomic patterns like 'd__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__'
            taxonomic_patterns = ['d__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
            self.otu_cols = []
            
            for col in self.df.columns:
                if col not in self.target_cols:  # Exclude target columns
                    # Check if column contains taxonomic patterns
                    if any(pattern in col for pattern in taxonomic_patterns):
                        self.otu_cols.append(col)
            
            self.feature_cols = self.otu_cols
            print(f"OTU mode: Identified {len(self.feature_cols)} OTU feature columns and {len(self.target_cols)} target columns")
            
            # Verify we have the expected number of OTU columns
            if len(self.feature_cols) != 1086:
                print(f"Warning: Expected 1086 OTU columns, found {len(self.feature_cols)}")
                # Show a few examples of feature column names for debugging
                print(f"Sample feature columns: {self.feature_cols[:5]}")
                if len(self.feature_cols) > 5:
                    print(f"... and {len(self.feature_cols) - 5} more")
                    
        elif self.graph_mode == 'family':
            # First identify OTU columns, then process at family level
            taxonomic_patterns = ['d__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
            self.otu_cols = []
            
            for col in self.df.columns:
                if col not in self.target_cols:  # Exclude target columns
                    # Check if column contains taxonomic patterns
                    if any(pattern in col for pattern in taxonomic_patterns):
                        self.otu_cols.append(col)
            
            print(f"Family mode: Found {len(self.otu_cols)} OTU columns, processing at family level...")
            
            # Process families and get filtered family columns
            self.df_family_filtered, self.feature_cols = self._process_families()
            
            print(f"Family mode: Selected {len(self.feature_cols)} family feature columns and {len(self.target_cols)} target columns")
        
        else:
            raise ValueError(f"Invalid graph_mode: {self.graph_mode}. Must be 'otu' or 'family'")
        
        print(f"Target columns: {self.target_cols}")
        
        # Extract and clean target data
        self.target_df = self.df[self.target_cols].copy()
        self.target_df.columns = self.target_df.columns.str.strip()
        
        # Clean and convert target data to numeric
        original_indices = self.target_df.index
        self.target_df = self._clean_target_data(self.target_df)
        
        # Update target column names after cleaning
        self.target_cols = list(self.target_df.columns)
        
        # If rows were removed during cleaning, update the main dataframe
        if len(self.target_df) != len(original_indices):
            self.df = self.df.loc[self.target_df.index]
        
        # Create node features
        self.df_features, self.feature_matrix = self._create_node_features()
        
        # Create initial graph structure using Mantel test
        self.full_edge_index, self.full_edge_weight, self.full_edge_type = self._create_graph_structure()
        
        # Create KNN sparsified graph structure (always use KNN for initial graph)
        self.edge_index, self.edge_weight, self.edge_type = self._create_knn_graph(k=k_neighbors)
        
        # Create PyG data objects
        self.data_list = self._create_data_objects()
        
        # Store original graph data for visualization
        self.original_graph_data = {
            'edge_index': self.edge_index.clone(),
            'edge_weight': self.edge_weight.clone(),
            'edge_type': self.edge_type.clone()
        }
        
        # Initialize explainer-sparsified graph data as None
        self.explainer_sparsified_graph_data = None
        
        # Create directory for visualizations
        os.makedirs('graph_visualizations', exist_ok=True)
    
    def _create_node_features(self):
        """Create node features from input data"""
        # Extract feature data based on graph mode
        if self.graph_mode == 'otu':
            df_features = self.df[self.feature_cols].copy()
        elif self.graph_mode == 'family':
            # Use the already filtered family data
            df_features = self.df_family_filtered.copy()
        else:
            raise ValueError(f"Invalid graph_mode: {self.graph_mode}")
        
        # Apply variance stabilization if needed
        # For microbial data, double square root transformation is common
        # Adjust this based on your data characteristics
        if df_features.min().min() >= 0:  # Check if all values are non-negative
            df_features = df_features.apply(lambda x: np.sqrt(np.sqrt(x + 1e-10)))
        
        # Convert to numpy array with shape [num_features, num_samples]
        feature_matrix = df_features.values.T.astype(np.float32)
        
        print(f"Node feature matrix: {feature_matrix.shape} (features × samples)")
        
        # Store the feature names for later use
        self.node_feature_names = list(df_features.columns)
        
        return df_features, feature_matrix
    
    def _compute_distance_matrix(self, vec, metric='euclidean'):
        """Compute distance matrix between samples for a given feature"""
        if vec.ndim == 1:
            vec = vec.reshape(-1, 1)
        dm = squareform(pdist(vec, metric=metric))
        return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _mantel_test(self, d1, d2, permutations=100):  # Reduced permutations for efficiency
        """Mantel test to assess correlation between two distance matrices"""
        n = d1.shape[0]
        idx = np.triu_indices(n, k=1)
        v1, v2 = d1[idx], d2[idx]
        
        if v1.std() == 0 or v2.std() == 0:
            return 1.0, 0.0
            
        r_obs = pearsonr(v1, v2)[0]
        
        # Reduced permutation test for efficiency
        count = 0
        for _ in range(permutations):
            perm_v2 = np.random.permutation(v2)
            r_perm = abs(pearsonr(v1, perm_v2)[0])
            if r_perm >= abs(r_obs):
                count += 1
                
        p_value = (count + 1) / (permutations + 1)
        return p_value, r_obs
    
    def _process_families(self):
        """Extract family level taxonomy and aggregate OTUs"""
        # Function to extract family from taxonomy string
        def extract_family(colname):
            for part in colname.split(';'):
                part = part.strip()
                if part.startswith('f__'):
                    return part[3:] or "UnclassifiedFamily"
            return "UnclassifiedFamily"
        
        # Map OTUs to families
        col_to_family = {c: extract_family(c) for c in self.otu_cols}
        family_to_cols = {}
        for c, fam in col_to_family.items():
            if fam not in family_to_cols:
                family_to_cols[fam] = []
            family_to_cols[fam].append(c)
        
        # Aggregate OTUs at family level
        df_fam = pd.DataFrame({
            fam: self.df[cols].sum(axis=1)
            for fam, cols in family_to_cols.items()
        }, index=self.df.index)
        
        # Convert to relative abundance
        df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
        
        print(f"Total families before filtering: {df_fam_rel.shape[1]}")
        
        # RELAXED FILTERING CRITERIA - Multiple options
        # Option 1: Very permissive (prevalence OR abundance)
        presence_count = (df_fam_rel > 0).sum(axis=0)
        prevalence = presence_count / df_fam_rel.shape[0]
        
        # More permissive thresholds
        prevalence_threshold = 0.02  # 2% of samples (≥1.08 samples)
        abundance_threshold = 0.001  # 0.1% mean abundance
        
        high_prev = prevalence[prevalence >= prevalence_threshold].index
        mean_abund = df_fam_rel.mean(axis=0)
        high_abund = mean_abund[mean_abund >= abundance_threshold].index
        
        # Use UNION instead of intersection to be more inclusive
        selected_families = high_prev.union(high_abund)
        
        # Alternative: If still too few, use even more permissive criteria
        if len(selected_families) < 50:  # Target at least 50 families
            print(f"Only {len(selected_families)} families with standard criteria. Using more permissive filtering...")
            
            # Ultra-permissive: families present in at least 1 sample with >0.05% abundance
            ultra_prev = prevalence[prevalence >= 0.018].index  # ~1 sample
            ultra_abund = mean_abund[mean_abund >= 0.0005].index  # 0.05% abundance
            selected_families = ultra_prev.union(ultra_abund)
        
        # Ensure we don't include completely absent families
        non_zero_families = df_fam_rel.columns[df_fam_rel.sum(axis=0) > 0]
        selected_families = selected_families.intersection(non_zero_families)
        
        df_fam_rel_filtered = df_fam_rel[selected_families].copy()
        
        print(f"Selected {len(selected_families)} families after relaxed filtering (out of {df_fam_rel.shape[1]}).")
        print(f"Prevalence threshold: {prevalence_threshold*100:.1f}% of samples")
        print(f"Abundance threshold: {abundance_threshold*100:.2f}% mean abundance")
        print(f"Filtering method: UNION (prevalence OR abundance)")
        
        # Show some statistics
        final_prevalence = (df_fam_rel_filtered > 0).sum(axis=0) / df_fam_rel_filtered.shape[0]
        final_abundance = df_fam_rel_filtered.mean(axis=0)
        
        print(f"Final prevalence range: {final_prevalence.min():.3f} - {final_prevalence.max():.3f}")
        print(f"Final abundance range: {final_abundance.min():.6f} - {final_abundance.max():.3f}")
        
        return df_fam_rel_filtered, list(df_fam_rel_filtered.columns)
    
    def _create_graph_structure(self):
        """Create graph structure based on correlation or distance metrics"""
        if self.graph_mode == 'family':
            # For family mode, we can optionally use biological distance method
            # For now, we'll use the biological distance method as default for family mode
            return self._create_biological_distance_graph_pyg()
        elif self.use_fast_correlation:
            return self._create_graph_structure_fast()
        else:
            return self._create_graph_structure_mantel()
    
    def _create_graph_structure_fast(self):
        """Create graph structure using correlation-based approach (much faster than Mantel tests)"""
        print("Constructing graph using fast correlation method...")
        
        num_features = len(self.node_feature_names)
        print(f"Computing correlations for {num_features} features...")
        
        # Use direct correlation approach which is much faster
        # Compute correlation matrix between features
        correlation_matrix = np.corrcoef(self.feature_matrix)
        
        # Replace NaN values with 0
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        edge_i, edge_j, edge_weights = [], [], []
        
        # Adaptive correlation threshold based on graph mode and feature count
        if self.graph_mode == 'family':
            # For family mode, use much stricter threshold due to small feature count
            if num_features <= 30:
                correlation_threshold = 0.7  # Very strict for small graphs
            else:
                correlation_threshold = 0.5
            print(f"Family mode: Using strict correlation threshold {correlation_threshold}")
        else:
            # For OTU mode, use moderate threshold
            correlation_threshold = 0.3
            print(f"OTU mode: Using correlation threshold {correlation_threshold}")
        
        # Create edges based on correlation strength
        for i in range(num_features):
            for j in range(i+1, num_features):
                corr = correlation_matrix[i, j]
                
                # Use absolute correlation above threshold
                if abs(corr) > correlation_threshold:
                    edge_i += [i, j]
                    edge_j += [j, i]
                    edge_weights += [abs(corr), abs(corr)]
        
        # If no edges meet the strict threshold, use top-k approach
        if len(edge_i) == 0:
            print(f"No edges meet threshold {correlation_threshold}, using top-k approach...")
            # Select top k pairs by correlation strength
            k = min(num_features * 2, 40)  # Adaptive k based on num_features
            
            # Get all correlation pairs
            corr_pairs = []
            for i in range(num_features):
                for j in range(i+1, num_features):
                    corr = abs(correlation_matrix[i, j])
                    if corr > 0.1:  # Minimum threshold to avoid noise
                        corr_pairs.append((i, j, corr))
            
            # Sort by correlation strength and take top k
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            corr_pairs = corr_pairs[:k]
            
            print(f"Selected top {len(corr_pairs)} edges by correlation strength")
            
            for i, j, corr in corr_pairs:
                edge_i += [i, j]
                edge_j += [j, i]
                edge_weights += [corr, corr]
        
        # Create edge types based on correlation sign
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            corr = correlation_matrix[i, j] if i < len(correlation_matrix) and j < len(correlation_matrix) else 0
            edge_type = 1 if corr > 0 else 0
            edge_types.append(edge_type)
        
        print(f"Created graph with {len(edge_i)//2} undirected edges using correlation method")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_graph_structure_mantel(self):
        """Create graph structure using Mantel test approach (slower but more rigorous)"""
        print("Constructing graph using Mantel test method (this may take a while)...")
        
        # Compute distance matrices for each feature
        dist_mats = {
            feat: self._compute_distance_matrix(self.feature_matrix[i], metric='euclidean')
            for i, feat in enumerate(self.node_feature_names)
        }
        
        # Construct edges based on Mantel test
        num_features = len(self.node_feature_names)
        edge_i, edge_j, edge_weights = [], [], []
        
        # Create edges between features with significant correlation
        for i in range(num_features):
            for j in range(i+1, num_features):
                # Use Mantel test to assess correlation
                p, r = self._mantel_test(
                    dist_mats[self.node_feature_names[i]],
                    dist_mats[self.node_feature_names[j]],
                    permutations=100  # Reduced for efficiency
                )
                
                if p < self.mantel_threshold:  # Significant correlation
                    edge_i += [i, j]
                    edge_j += [j, i]
                    edge_weights += [abs(r), abs(r)]  # Use correlation strength as edge weight
        
        # Create edge types based on correlation sign
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            # Calculate correlation coefficient between the two features
            corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
            # Edge type: 0 = negative correlation, 1 = positive correlation
            edge_type = 1 if corr > 0 else 0
            edge_types.append(edge_type)
        
        print(f"Created graph with {len(edge_i)//2} undirected edges using Mantel test")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_biological_distance_graph(self):
        """Create graph using biological distance metrics for family-level data"""
        print("Constructing graph using biological distance method for family mode...")
        
        # For family-level data, we can use phylogenetic/taxonomic distance
        # This creates a more biologically meaningful graph structure
        
        num_features = len(self.node_feature_names)
        print(f"Creating biological distance graph for {num_features} families...")
        
        # Create distance matrix based on abundance profiles
        feature_matrix = self.feature_matrix
        
        # Method 1: Jensen-Shannon divergence (good for compositional data)
        edge_i, edge_j, edge_weights = [], [], []
        
        for i in range(num_features):
            for j in range(i+1, num_features):
                # Calculate Jensen-Shannon distance
                profile_i = feature_matrix[i] + 1e-10  # Add small epsilon
                profile_j = feature_matrix[j] + 1e-10
                
                # Normalize to probability distributions
                profile_i_norm = profile_i / profile_i.sum()
                profile_j_norm = profile_j / profile_j.sum()
                
                # Calculate JS distance (lower = more similar)
                js_distance = jensenshannon(profile_i_norm, profile_j_norm)
                
                # Convert distance to similarity (higher = more similar)
                similarity = 1.0 / (1.0 + js_distance)
                
                # Only keep edges above threshold
                threshold = 0.7  # Similarity threshold
                if similarity > threshold:
                    edge_i.extend([i, j])
                    edge_j.extend([j, i])
                    edge_weights.extend([similarity, similarity])
        
        # If no edges meet threshold, use top-k approach
        if len(edge_i) == 0:
            print("No edges meet biological distance threshold, using top-k approach...")
            k_per_node = min(3, num_features - 1)  # Each node connects to top 3 most similar
            
            for i in range(num_features):
                similarities = []
                for j in range(num_features):
                    if i != j:
                        profile_i = feature_matrix[i] + 1e-10
                        profile_j = feature_matrix[j] + 1e-10
                        
                        profile_i_norm = profile_i / profile_i.sum()
                        profile_j_norm = profile_j / profile_j.sum()
                        
                        js_distance = jensenshannon(profile_i_norm, profile_j_norm)
                        similarity = 1.0 / (1.0 + js_distance)
                        similarities.append((j, similarity))
                
                # Sort by similarity and take top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                for j, sim in similarities[:k_per_node]:
                    edge_i.append(i)
                    edge_j.append(j)
                    edge_weights.append(sim)
        
        # Create edge type based on co-occurrence patterns
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            # Simple rule: positive correlation → type 1, negative → type 0
            corr = np.corrcoef(feature_matrix[i], feature_matrix[j])[0, 1]
            edge_types.append(1 if corr > 0 else 0)
        
        print(f"Created biological distance graph with {len(edge_i)} directed edges")
        
        return edge_i, edge_j, edge_weights, edge_types
    
    def _create_biological_distance_graph_pyg(self):
        """Create graph using biological distance metrics and return PyG format"""
        edge_i, edge_j, edge_weights, edge_types = self._create_biological_distance_graph()
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_knn_graph(self, k=None):
        """Create a k-nearest neighbor sparsified version of the graph"""
        if k is None:
            k = self.k_neighbors
            
        print(f"Creating KNN graph with k={k}...")
        
        # Create adjacency matrix from full edge_index
        num_nodes = len(self.node_feature_names)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(self.full_edge_index.shape[1]):
            u, v = self.full_edge_index[0, i], self.full_edge_index[1, i]
            adj_matrix[u, v] = self.full_edge_weight[i]
        
        # KNN sparsification
        adj_matrix_np = adj_matrix.numpy()
        
        # For each node, keep only the k strongest connections
        for i in range(num_nodes):
            # Get weights of all neighbors
            neighbors = adj_matrix_np[i]
            
            # Sort neighbors by weight and keep only top k
            if np.sum(neighbors > 0) > k:
                threshold = np.sort(neighbors)[-k]
                adj_matrix_np[i, neighbors < threshold] = 0
        
        # Make matrix symmetric (undirected graph)
        adj_matrix_np = np.maximum(adj_matrix_np, adj_matrix_np.T)
        
        # Convert back to edge_index and edge_weight format
        new_edge_index = []
        new_edge_weight = []
        new_edge_type = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix_np[i, j] > 0:
                    new_edge_index.append([i, j])
                    new_edge_weight.append(adj_matrix_np[i, j])
                    
                    # Determine edge type (sign of correlation)
                    corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
                    new_edge_type.append(1 if corr > 0 else 0)
        
        new_edge_index = torch.tensor(new_edge_index).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
        
        print(f"KNN graph has {new_edge_index.shape[1]//2} undirected edges")
        
        return new_edge_index, new_edge_weight, new_edge_type
    
    def _create_data_objects(self):
        """Create PyG Data objects for each sample"""
        # Transpose feature matrix to shape [num_samples, num_features]
        feature_matrix_samples = self.feature_matrix.T
        
        # Create a list of PyG Data objects
        data_list = []
        
        for s in range(feature_matrix_samples.shape[0]):
            # Node features for this sample - feature values
            x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
            
            # Graph-level targets
            targets = torch.tensor(self.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
            
            # Create the Data object
            data = Data(
                x=x,
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                edge_attr=self.edge_weight.view(-1, 1),
                edge_type=self.edge_type,
                y=targets
            )
            
            data_list.append(data)
        
        print(f"Created {len(data_list)} graph data objects with {len(self.node_feature_names)} nodes each")
        
        return data_list
    
    def visualize_graphs(self, save_dir='graph_visualizations'):
        """Visualize both original and sparsified graphs for comparison"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Get number of nodes
        num_nodes = len(self.node_feature_names)
        
        # Visualize original graph
        self._visualize_single_graph(
            self.original_graph_data['edge_index'],
            self.original_graph_data['edge_weight'],
            self.original_graph_data['edge_type'],
            ax1,
            title=f"KNN Graph ({num_nodes} nodes, {self.original_graph_data['edge_index'].shape[1]//2} edges)"
        )
        
        # Check if explainer-sparsified graph exists
        if self.explainer_sparsified_graph_data is not None:
            # Visualize explainer-sparsified graph
            self._visualize_single_graph(
                self.explainer_sparsified_graph_data['edge_index'],
                self.explainer_sparsified_graph_data['edge_weight'],
                self.explainer_sparsified_graph_data['edge_type'],
                ax2,
                title=f"GNNExplainer Graph ({num_nodes} nodes, {self.explainer_sparsified_graph_data['edge_index'].shape[1]//2} edges)"
            )
        else:
            ax2.text(0.5, 0.5, "GNNExplainer graph not created yet.",
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax2.set_title("GNNExplainer Graph (Not Available)")
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/graph_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {save_dir}/graph_comparison.png")
        
        # Also create individual high-resolution visualizations
        plt.figure(figsize=(15, 15))
        self._visualize_single_graph(
            self.original_graph_data['edge_index'],
            self.original_graph_data['edge_weight'],
            self.original_graph_data['edge_type'],
            plt.gca(),
            title=f"KNN Graph ({num_nodes} nodes, {self.original_graph_data['edge_index'].shape[1]//2} edges)"
        )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/knn_graph.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.explainer_sparsified_graph_data is not None:
            plt.figure(figsize=(15, 15))
            self._visualize_single_graph(
                self.explainer_sparsified_graph_data['edge_index'],
                self.explainer_sparsified_graph_data['edge_weight'],
                self.explainer_sparsified_graph_data['edge_type'],
                plt.gca(),
                title=f"GNNExplainer Graph ({num_nodes} nodes, {self.explainer_sparsified_graph_data['edge_index'].shape[1]//2} edges)"
            )
            plt.tight_layout()
            plt.savefig(f"{save_dir}/explainer_graph.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_single_graph(self, edge_index, edge_weight, edge_type, ax, title):
        """Helper method to visualize a single graph"""
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(self.node_feature_names):
            G.add_node(i, name=name)
        
        # Add edges with weights and types
        for i in range(0, edge_index.shape[1], 2):  # Process only one direction for undirected edges
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weight[i].item()
            edge_t = edge_type[i].item()
            G.add_edge(u, v, weight=weight, type=edge_t)
        
        # Create layout
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Calculate node size based on degree centrality
        node_size = []
        for node in G.nodes():
            # Use degree centrality
            degree = G.degree(node, weight='weight')
            node_size.append(100 + 500 * degree)
        
        # Scale edge width by correlation strength and color by type
        edge_colors = []
        edge_width = []
        
        for u, v, data in G.edges(data=True):
            # Edge type determines color: 0 = negative correlation, 1 = positive correlation
            if data['type'] == 0:
                edge_colors.append('red')  # negative correlation
            else:
                edge_colors.append('green')  # positive correlation
            
            # Width based on weight
            edge_width.append(abs(data['weight']) * 2 + 0.5)
        
        # Try to find communities for node coloring
        try:
            from community import best_partition
            partition = best_partition(G)
            node_colors = [partition[node] for node in G.nodes()]
        except:
            # Fallback if community detection fails
            node_colors = list(range(len(G.nodes)))
        
        # Draw the graph
        nx.draw_networkx(
            G, 
            pos=pos,
            with_labels=True,
            labels={node: self.node_feature_names[node] for node in G.nodes()},
            node_size=node_size,
            node_color=node_colors,
            width=edge_width,
            edge_color=edge_colors,
            alpha=0.8,
            cmap=plt.cm.tab20,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # Create a legend for edge types
        ax.plot([], [], 'g-', linewidth=2, label='Positive correlation')
        ax.plot([], [], 'r-', linewidth=2, label='Negative correlation')
        ax.legend(loc='upper right')
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
    
    def _clean_target_data(self, target_df):
        """Clean and convert target data to numeric format"""
        print("Cleaning target data...")
        print(f"Target columns before cleaning: {list(target_df.columns)}")
        
        # Convert all columns to numeric, replacing non-numeric values with NaN
        for col in target_df.columns:
            target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
        
        # Check for completely missing targets
        missing_data_summary = target_df.isna().sum()
        print(f"Missing values per target: {missing_data_summary.to_dict()}")
        
        # Check for rows with all NaN values and remove them
        all_nan_rows = target_df.isna().all(axis=1)
        if all_nan_rows.any():
            print(f"Removing {all_nan_rows.sum()} samples with all NaN target values")
            target_df = target_df[~all_nan_rows]
        
        # Fill remaining NaN values with column means
        target_df = target_df.fillna(target_df.mean())
        
        # If there are still NaN values (columns with all NaN), fill with 0
        target_df = target_df.fillna(0)
        
        # Convert all columns to float32 to ensure consistent dtypes for PyTorch
        target_df = target_df.astype(np.float32)
        
        print(f"Cleaned target data: {target_df.shape[1]} targets, {target_df.shape[0]} samples")
        print(f"Target data types: {target_df.dtypes.value_counts().to_dict()}")
        print(f"Target value ranges:")
        for col in target_df.columns:
            print(f"  {col}: min={target_df[col].min():.3f}, max={target_df[col].max():.3f}, mean={target_df[col].mean():.3f}")
        
        return target_df 