import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx

# Import custom modules
from MethaneGNN_dataset import MethaneGNNDataset
from MethaneGNN_models import MethaneGNN, MethaneGNNExplainer, simple_GCN_res, simple_RGGC, simple_GAT

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MethanePipeline:
    """Complete pipeline for methane prediction with graph sparsification and GNN explanation"""
    
    def __init__(self, 
                 data_path='../Data/New_data.csv',
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 model_type='gat',
                 model_architecture='default',  # Added parameter for model architecture
                 hidden_dim=64,
                 num_layers=4,
                 dropout_rate=0.3,
                 batch_size=8,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 num_epochs=300,
                 patience=30,
                 num_folds=5,
                 save_dir='./methane_results',
                 visualize_graphs=True):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the CSV file with microbial data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
            model_type: Type of GNN model ('gcn', 'gat', 'gatv2', 'gin')
            model_architecture: Architecture to use ('default', 'simple_gcn_res', 'simple_rggc', 'simple_gat')
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout_rate: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
            num_folds: Number of folds for cross-validation
            save_dir: Directory to save results
            visualize_graphs: Whether to visualize the graphs
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_folds = num_folds
        self.save_dir = save_dir
        self.visualize_graphs = visualize_graphs
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations/ACE-km", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations/H2-km", exist_ok=True)
        
        # Target names
        self.target_names = ['ACE-km', 'H2-km']
        
        # Load and process data
        self.dataset = MethaneGNNDataset(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold
        )
        
        # Create KNN sparsified graph
        self.sparsified_data = self.dataset.create_knn_sparsified_graph(k=k_neighbors)
        
        # Visualize the graphs if requested
        if self.visualize_graphs:
            print("\nVisualizing original and sparsified graphs...")
            self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
            print(f"Graph visualizations saved to {self.save_dir}/graphs/")
        
        # Store the data
        self.data_list = self.sparsified_data
        self.node_feature_names = self.dataset.node_feature_names
    
    def _create_model(self, num_targets=1):
        """Create model based on specified architecture and type"""
        if self.model_architecture == 'default':
            model = MethaneGNN(
                num_node_features=1,
                hidden_dim=self.hidden_dim,
                num_targets=num_targets,
                dropout_rate=self.dropout_rate,
                num_layers=self.num_layers,
                model_type=self.model_type
            ).to(device)
        elif self.model_architecture == 'simple_gcn_res':
            model = simple_GCN_res(
                hidden_channels=self.hidden_dim,
                n_classes=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1
            ).to(device)
        elif self.model_architecture == 'simple_rggc':
            model = simple_RGGC(
                hidden_channels=self.hidden_dim,
                n_classes=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1
            ).to(device)
        elif self.model_architecture == 'simple_gat':
            model = simple_GAT(
                hidden_channels=self.hidden_dim,
                n_classes=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                num_heads=1
            ).to(device)
        else:
            raise ValueError(f"Unknown model architecture: {self.model_architecture}")
        
        return model
    
    def train_single_target(self, target_idx, target_name):
        """Train a model for a single target variable"""
        print(f"\n{'='*50}")
        print(f"Training model for {target_name}")
        print(f"{'='*50}")
        
        # Prepare data with single target
        single_target_data_list = []
        for data in self.data_list:
            # Create a copy with only the specific target
            target_value = data.y[0, target_idx].item()
            single_target_data = data.clone()
            single_target_data.y = torch.tensor([[target_value]], dtype=torch.float32)
            single_target_data_list.append(single_target_data)
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        
        # Iterate through folds
        for fold, (train_index, test_index) in enumerate(kf.split(single_target_data_list)):
            fold_num = fold + 1
            print(f"Fold {fold_num}: Train on {len(train_index)} samples, Test on {len(test_index)} samples")
            
            # Split into train and test sets
            train_dataset = [single_target_data_list[i] for i in train_index]
            test_dataset = [single_target_data_list[i] for i in test_index]
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model using the helper method
            model = self._create_model(num_targets=1)
            
            # Setup optimizer and loss function
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(1, self.num_epochs+1):
                # Training step
                model.train()
                total_loss = 0
                
                for batch_data in train_loader:
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    out = model(batch_data)
                    # Reshape target to match output
                    target = batch_data.y.view(-1, 1)
                    
                    # Check if the model returns uncertainty estimates (tuple)
                    if isinstance(out, tuple):
                        # Unpack predictions and log variance
                        pred, log_var = out
                        
                        # Implement uncertainty-aware loss 
                        # L = (pred - target)^2 / (2 * exp(log_var)) + log_var/2
                        # This is negative log-likelihood of Gaussian with predicted mean and variance
                        precision = torch.exp(-log_var)
                        loss = torch.mean(precision * (pred - target) ** 2 + log_var)
                    else:
                        # Standard MSE loss if no uncertainty estimates
                        loss = criterion(out, target)
                    
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch_data.num_graphs
                
                avg_loss = total_loss / len(train_dataset)
                train_losses.append(avg_loss)
                
                # Evaluate on test set
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        pred = model(batch_data)
                        # Handle tuple output from model (only take predictions)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        
                        # Reshape target
                        target = batch_data.y.view(-1, 1)
                        val_loss += criterion(pred, target).item() * batch_data.num_graphs
                    val_loss /= len(test_dataset)
                    val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 20 == 0 or epoch == self.num_epochs:
                    print(f"Epoch {epoch:03d} - Train MSE: {avg_loss:.4f}, Val MSE: {val_loss:.4f}")
            
            # Load best model for this fold
            model.load_state_dict(best_model_state)
            
            # Final evaluation on test set
            model.eval()
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    pred = model(batch_data)
                    # Handle tuple output from model
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    
                    target = batch_data.y.view(-1, 1)
                    all_preds.append(pred.cpu().numpy())
                    all_trues.append(target.cpu().numpy())
            
            all_preds = np.vstack(all_preds)
            all_trues = np.vstack(all_trues)
            fold_mse = mean_squared_error(all_trues, all_preds)
            fold_rmse = np.sqrt(fold_mse)
            fold_r2 = r2_score(all_trues, all_preds)
            
            print(f"Fold {fold_num} results: MSE = {fold_mse:.4f}, RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")
            
            # Store fold results
            fold_results.append({
                'fold': fold_num,
                'preds': all_preds,
                'trues': all_trues,
                'mse': fold_mse,
                'rmse': fold_rmse,
                'r2': fold_r2
            })
            
            # Save the best model for this fold
            torch.save(model.state_dict(), f"{self.save_dir}/models/{target_name}_model_fold{fold_num}.pt")
            
            # Create plot for this fold
            plt.figure(figsize=(8, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title(f'{target_name} - Fold {fold_num} Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.savefig(f"{self.save_dir}/plots/{target_name}_loss_fold{fold_num}.png")
            plt.close()
            
            # Generate explanations for test samples
            self._generate_explanations(model, test_dataset, target_name, fold_num, target_idx)
        
        # Train final model on all data
        # Commenting out final model training on all data
        # self._train_final_model(single_target_data_list, target_name, target_idx)
        
        # Calculate and display overall performance
        self._evaluate_performance(fold_results, target_name)
        
        return fold_results
    
    def _train_final_model(self, data_list, target_name, target_idx):
        """Train a final model on all data"""
        # Commenting out the entire final model training method
        """
        print(f"\nTraining final {target_name} model on all data...")
        
        # Create dataloader with all data
        all_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        final_model = self._create_model(num_targets=1)
        
        # Setup optimizer and loss function
        optimizer = Adam(final_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-5
        )
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(1, self.num_epochs+1):
            final_model.train()
            total_loss = 0
            
            for batch_data in all_loader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                out = final_model(batch_data)
                target = batch_data.y.view(-1, 1)
                
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_data.num_graphs
            
            # Update learning rate
            scheduler.step()
            
            if epoch % 20 == 0 or epoch == self.num_epochs:
                print(f"Epoch {epoch:03d} - Train MSE: {total_loss/len(all_loader):.4f}")
        
        # Save the final model
        torch.save(final_model.state_dict(), f"{self.save_dir}/models/{target_name}_final_model.pt")
        print(f"Final {target_name} model saved to {self.save_dir}/models/{target_name}_final_model.pt")
        
        # Generate explanations using the final model
        self._generate_explanations(final_model, data_list[:10], target_name, 'final', target_idx)
        """
        pass
    
    def _generate_explanations(self, model, dataset, target_name, fold, target_idx):
        """Generate and save GNN explanations"""
        # Initialize explainer
        explainer = MethaneGNNExplainer(model, device)
        
        # Create output directory
        os.makedirs(f"{self.save_dir}/explanations/{target_name}/fold_{fold}", exist_ok=True)
        
        # Generate explanations for each sample
        for i, data in enumerate(dataset[:5]):  # Limit to first 5 samples to save time
            # Generate explanation
            edge_importance_matrix, explanation = explainer.explain_graph(
                data, 
                node_names=self.node_feature_names,
                save_path=f"{self.save_dir}/explanations/{target_name}/fold_{fold}/sample_{i}_edge_importance.csv",
                target_idx=0  # For single target models, target_idx is always 0
            )
            
            # Visualize the explanation as a network
            self._visualize_explanation(
                data, 
                edge_importance_matrix, 
                i, 
                f"{self.save_dir}/explanations/{target_name}/fold_{fold}/sample_{i}_explanation.png",
                target_name
            )
    
    def _visualize_explanation(self, data, edge_importance_matrix, sample_id, save_path, target_name):
        """Visualize explanation as a network with edge importance"""
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(self.node_feature_names):
            G.add_node(i, name=name, abundance=data.x[i].item())
        
        # Add edges with importance as weights - use a higher threshold to reduce disconnected components
        edge_importance = edge_importance_matrix.cpu().numpy()
        
        # Normalize edge importance to [0, 1] range
        if edge_importance.max() > 0:
            edge_importance = edge_importance / edge_importance.max()
        
        # Add edges with importance above a threshold
        threshold = 0.2  # Only keep edges with at least 20% of max importance
        for i in range(len(self.node_feature_names)):
            for j in range(i+1, len(self.node_feature_names)):
                importance = edge_importance[i, j]
                if importance > threshold:
                    G.add_edge(i, j, weight=float(importance))
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Create layout
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Get node sizes based on abundance
        node_size = [1000 * (0.1 + data.x[i].item()) for i in range(len(self.node_feature_names))]
        
        # Get edge widths based on importance
        edge_width = []
        for u, v, d in G.edges(data=True):
            edge_width.append(d['weight'] * 5)
        
        # Draw the network
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            labels={i: self.node_feature_names[i] for i in G.nodes()},
            node_size=node_size,
            width=edge_width,
            edge_color='gray',
            font_size=8,
            font_weight='bold',
            alpha=0.8
        )
        
        plt.title(f'{target_name} Explanation - Sample {sample_id}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_performance(self, fold_results, target_name):
        """Calculate and display overall performance metrics"""
        # Combine predictions from all folds
        all_preds = np.vstack([fold['preds'] for fold in fold_results])
        all_trues = np.vstack([fold['trues'] for fold in fold_results])
        
        # Calculate metrics
        mse = mean_squared_error(all_trues, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_trues, all_preds)
        
        print(f"\n{target_name} Cross-Validation Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(all_trues, all_preds, alpha=0.6, edgecolor='k')
        
        # Set axis limits based on the target variable
        if target_name == 'ACE-km':
            plt.xlim(0, 50)
            plt.ylim(0, 50)
        elif target_name == 'H2-km':
            plt.xlim(0, 140)
            plt.ylim(0, 140)
        
        # Add 45-degree line
        plt.plot([0, plt.xlim()[1]], [0, plt.ylim()[1]], 'r--')
        
        plt.title(f'{target_name}\nMSE={mse:.4f}, R²={r2:.4f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.save_dir}/plots/{target_name}_overall_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            'Target': target_name,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }])
        metrics_df.to_csv(f"{self.save_dir}/plots/{target_name}_metrics.csv", index=False)
    
    def run_pipeline(self):
        """Run the complete pipeline for all targets"""
        results = {}
        
        # Train separate model for each target
        for target_idx, target_name in enumerate(self.target_names):
            results[target_name] = self.train_single_target(target_idx, target_name)
        
        # Create combined plot for both targets
        plt.figure(figsize=(15, 7))
        
        for i, target_name in enumerate(self.target_names):
            # Get data
            fold_results = results[target_name]
            all_preds = np.vstack([fold['preds'] for fold in fold_results])
            all_trues = np.vstack([fold['trues'] for fold in fold_results])
            
            # Calculate metrics
            mse = mean_squared_error(all_trues, all_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(all_trues, all_preds)
            
            # Create subplot
            plt.subplot(1, 2, i+1)
            plt.scatter(all_trues, all_preds, alpha=0.6, edgecolor='k')
            
            # Set axis limits based on the target variable
            if target_name == 'ACE-km':
                plt.xlim(0, 50)
                plt.ylim(0, 50)
            elif target_name == 'H2-km':
                plt.xlim(0, 140)
                plt.ylim(0, 140)
            
            # Add 45-degree line
            plt.plot([0, plt.xlim()[1]], [0, plt.ylim()[1]], 'r--')
            
            plt.title(f'{target_name}\nMSE={mse:.4f}, R²={r2:.4f}')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/combined_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to {self.save_dir}")
        
        return results


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = MethanePipeline(
        data_path='../Data/New_data.csv',
        k_neighbors=10,
        mantel_threshold=0.05,
        model_type='gin',
        model_architecture='default',
        hidden_dim=128,
        num_layers=4,
        dropout_rate=0.3,
        batch_size=8,
        learning_rate=0.001,
        weight_decay=1e-4,
        num_epochs=300,
        patience=30,
        num_folds=5,
        save_dir='./methane_results',
        visualize_graphs=True
    )
    
    results = pipeline.run_pipeline() 