import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data

# Import other modules
from dataset_regression import MicrobialGNNDataset
from explainer_regression import GNNExplainerRegression
from utils_regression import *
from pipeline_training import train_model
from pipeline_explainer import create_explainer_sparsified_graph

# Import the models
from GNNmodelsRegression import (
    simple_GCN_res_regression,
    simple_GCN_res_plus_regression,
    simple_RGGC_regression,
    simple_RGGC_plus_regression,
    simple_GAT_regression,
    GaussianNLLLoss
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
DEFAULT_SAVE_DIR = './regression_results'

class RegressionPipeline:
    """Complete pipeline for graph-based regression with GNN models"""
    
    def __init__(self, 
                 data_path,
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 model_type='gcn',
                 hidden_dim=64,
                 dropout_rate=0.3,
                 batch_size=8,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 num_epochs=200,
                 patience=20,
                 num_folds=5,
                 save_dir=DEFAULT_SAVE_DIR,
                 importance_threshold=0.3,
                 estimate_uncertainty=False,
                 use_fast_correlation=True,
                 graph_mode='otu'):
        """
        Initialize the regression pipeline
        
        Args:
            data_path: Path to the CSV file with data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
            model_type: Type of GNN model ('gcn', 'gat', 'rggc')
            hidden_dim: Hidden dimension size for GNN
            dropout_rate: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
            num_folds: Number of folds for cross-validation
            save_dir: Directory to save results
            importance_threshold: Threshold for edge importance in GNNExplainer sparsification
            estimate_uncertainty: Whether to estimate uncertainty in predictions
            use_fast_correlation: If True, use fast correlation-based graph construction
            graph_mode: Mode for graph construction ('otu' or 'family')
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_folds = num_folds
        self.save_dir = save_dir
        self.importance_threshold = importance_threshold
        self.estimate_uncertainty = estimate_uncertainty
        self.use_fast_correlation = use_fast_correlation
        self.graph_mode = graph_mode
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations", exist_ok=True)
        
        # Load and process data
        print("\nLoading and processing data...")
        self.dataset = MicrobialGNNDataset(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode
        )
        
        # Get target names for reference
        self.target_names = self.dataset.target_cols 

    def create_model(self, num_targets=1):
        """Create a GNN model based on specified type"""
        if self.model_type == 'gcn':
            model = simple_GCN_res_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        elif self.model_type == 'rggc':
            model = simple_RGGC_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        elif self.model_type == 'gat':
            model = simple_GAT_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                num_heads=4,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train_model(self, target_idx=None, data_list=None):
        """
        Train a GNN model for regression
        
        Args:
            target_idx: Index of the target variable to predict (if None, predict all targets)
            data_list: List of graph data objects to use (if None, use self.dataset.data_list)
            
        Returns:
            Dictionary with training results
        """
        return train_model(self, target_idx, data_list)
    
    def create_explainer_sparsified_graph(self, model, target_idx=0, importance_threshold=None):
        """
        Create a sparsified graph based on GNNExplainer results
        
        Args:
            model: Trained GNN model
            target_idx: Index of the target variable to explain
            importance_threshold: Threshold for edge importance
            
        Returns:
            List of sparsified graph data objects
        """
        if importance_threshold is None:
            importance_threshold = self.importance_threshold
            
        return create_explainer_sparsified_graph(self, model, target_idx, importance_threshold)

    def run_pipeline(self):
        """
        Run the complete regression pipeline:
        1. Train models on KNN-sparsified graph (created during initialization)
        2. Create further sparsified graph using GNNExplainer
        3. Train models on GNNExplainer-sparsified graph
        4. Compare and analyze results
        
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Step 1: Train models on KNN-sparsified graph (already created during initialization)
        print("\n" + "="*80)
        print("STEP 1: Training on KNN-sparsified graph")
        print("="*80)
        
        knn_results = {}
        
        if len(self.target_names) > 1:
            # Either train a single multi-target model or separate models for each target
            train_separate = True  # Set to False if you want to train a multi-target model
            
            if train_separate:
                for target_idx, target_name in enumerate(self.target_names):
                    print(f"\nTraining for target: {target_name}")
                    knn_results[target_name] = self.train_model(target_idx=target_idx)
            else:
                print("\nTraining multi-target model")
                knn_results['all_targets'] = self.train_model()
        else:
            # Only one target, train a single model
            target_name = self.target_names[0]
            print(f"\nTraining for target: {target_name}")
            knn_results[target_name] = self.train_model(target_idx=0)
        
        # Step 2: Create GNNExplainer-sparsified graph
        print("\n" + "="*80)
        print("STEP 2: Creating GNNExplainer-sparsified graph")
        print("="*80)
        
        # Choose a model from the first target's results to use for explanation
        if len(self.target_names) > 1:
            target_name = self.target_names[0]
            model = knn_results[target_name][0]['model']  # Use first fold model
            target_idx = 0
        else:
            target_name = self.target_names[0]
            model = knn_results[target_name][0]['model']
            target_idx = 0
        
        # Create sparsified graph data using GNNExplainer
        explainer_sparsified_data = self.create_explainer_sparsified_graph(
            model, 
            target_idx=target_idx, 
            importance_threshold=self.importance_threshold
        )
        
        # Visualize the graphs
        self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
        # Step 3: Train models on GNNExplainer-sparsified graph
        print("\n" + "="*80)
        print("STEP 3: Training on GNNExplainer-sparsified graph")
        print("="*80)
        
        explainer_results = {}
        
        if len(self.target_names) > 1:
            # Either train a single multi-target model or separate models for each target
            train_separate = True  # Set to False if you want to train a multi-target model
            
            if train_separate:
                for target_idx, target_name in enumerate(self.target_names):
                    print(f"\nTraining for target: {target_name} (GNNExplainer-sparsified graph)")
                    explainer_results[target_name] = self.train_model(target_idx=target_idx, data_list=explainer_sparsified_data)
            else:
                print("\nTraining multi-target model (GNNExplainer-sparsified graph)")
                explainer_results['all_targets'] = self.train_model(data_list=explainer_sparsified_data)
        else:
            # Only one target, train a single model
            target_name = self.target_names[0]
            print(f"\nTraining for target: {target_name} (GNNExplainer-sparsified graph)")
            explainer_results[target_name] = self.train_model(target_idx=0, data_list=explainer_sparsified_data)
        
        # Step 4: Compare results
        print("\n" + "="*80)
        print("STEP 4: Comparing results between KNN and GNNExplainer-sparsified graphs")
        print("="*80)
        
        comparison_results = self._compare_results(knn_results, explainer_results)
        
        # Return consolidated results
        return {
            'knn': knn_results,
            'explainer': explainer_results,
            'comparison': comparison_results
        }
    
    def _compare_results(self, knn_results, explainer_results):
        """Compare results between KNN and GNNExplainer graphs"""
        comparison_results = {}
        
        for target_name in self.target_names:
            if target_name in knn_results and target_name in explainer_results:
                # Extract metrics
                knn_metrics = []
                explainer_metrics = []
                
                for fold_result in knn_results[target_name]:
                    for metric in fold_result['metrics']:
                        if metric['target_name'] == target_name:
                            knn_metrics.append(metric)
                
                for fold_result in explainer_results[target_name]:
                    for metric in fold_result['metrics']:
                        if metric['target_name'] == target_name:
                            explainer_metrics.append(metric)
                
                # Calculate average metrics
                knn_mse = np.mean([m['mse'] for m in knn_metrics])
                knn_rmse = np.mean([m['rmse'] for m in knn_metrics])
                knn_r2 = np.mean([m['r2'] for m in knn_metrics])
                
                explainer_mse = np.mean([m['mse'] for m in explainer_metrics])
                explainer_rmse = np.mean([m['rmse'] for m in explainer_metrics])
                explainer_r2 = np.mean([m['r2'] for m in explainer_metrics])
                
                # Calculate improvement
                mse_improvement = (knn_mse - explainer_mse) / knn_mse * 100
                rmse_improvement = (knn_rmse - explainer_rmse) / knn_rmse * 100
                r2_improvement = (explainer_r2 - knn_r2) / abs(knn_r2) * 100 if knn_r2 != 0 else 0
                
                # Store results
                comparison_results[target_name] = {
                    'knn': {
                        'mse': knn_mse,
                        'rmse': knn_rmse,
                        'r2': knn_r2
                    },
                    'explainer': {
                        'mse': explainer_mse,
                        'rmse': explainer_rmse,
                        'r2': explainer_r2
                    },
                    'improvement': {
                        'mse': mse_improvement,
                        'rmse': rmse_improvement,
                        'r2': r2_improvement
                    }
                }
                
                print(f"\nResults for target: {target_name}")
                print(f"KNN Graph - MSE: {knn_mse:.4f}, RMSE: {knn_rmse:.4f}, R²: {knn_r2:.4f}")
                print(f"GNNExplainer Graph - MSE: {explainer_mse:.4f}, RMSE: {explainer_rmse:.4f}, R²: {explainer_r2:.4f}")
                print(f"Improvement - MSE: {mse_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%")
        
        # Create comparison plot
        plot_comparison_results(comparison_results, f"{self.save_dir}/comparison_plot.png")
        
        # Save comparison results
        comparison_df = []
        
        for target_name, results in comparison_results.items():
            comparison_df.append({
                'target': target_name,
                'knn_mse': results['knn']['mse'],
                'knn_rmse': results['knn']['rmse'],
                'knn_r2': results['knn']['r2'],
                'explainer_mse': results['explainer']['mse'],
                'explainer_rmse': results['explainer']['rmse'],
                'explainer_r2': results['explainer']['r2'],
                'mse_improvement': results['improvement']['mse'],
                'rmse_improvement': results['improvement']['rmse'],
                'r2_improvement': results['improvement']['r2']
            })
        
        # Convert to DataFrame and save
        comparison_df = pd.DataFrame(comparison_df)
        comparison_df.to_csv(f"{self.save_dir}/comparison_results.csv", index=False)
        
        return comparison_results 