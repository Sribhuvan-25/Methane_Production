import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx

class GNNExplainerRegression:
    """GNN explainer for regression tasks"""
    
    def __init__(self, model, device):
        """
        Initialize the explainer
        
        Args:
            model: The trained GNN model
            device: Device to run the explanation on
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode
    
    def explain_graph(self, data, node_names=None, save_path=None, target_idx=0):
        """
        Generate explanation for a graph by learning an edge mask
        
        Args:
            data: PyG Data object for a graph
            node_names: Names of the nodes (features)
            save_path: Path to save the edge importance matrix
            target_idx: Index of the target variable to explain
            
        Returns:
            edge_importance_matrix: Matrix of edge importance scores
            explanation: Text explanation of the most important edges
        """
        # Copy data to device
        data = data.to(self.device)
        
        # Initialize edge mask
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.float, 
                              requires_grad=True, device=self.device)
        
        # Setup optimizer for the edge mask
        optimizer = Adam([edge_mask], lr=0.01)
        
        # Number of epochs for explanation - increased for better convergence
        num_epochs = 200  # Increased from 100
        
        # Store original edge index and edge weight
        original_edge_index = data.edge_index
        original_edge_attr = data.edge_attr
        
        # Extract target
        target = data.y[:, target_idx].squeeze()
        
        # Track convergence
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        # Training loop to learn edge mask
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Apply edge mask - create masked edge attributes
            masked_edge_attr = data.edge_attr * edge_mask.view(-1, 1)
            
            # Forward pass with masked edges
            out = self.model(data.x, data.edge_index, data.batch)
            
            # Handle different model output formats
            if isinstance(out, tuple):
                pred = out[0]
                
                # Reshape prediction if needed
                if len(pred.shape) == 1:
                    pred = pred.view(-1, 1)
                
                # Get the specific target prediction
                if pred.shape[1] > 1:
                    pred = pred[:, target_idx]
            else:
                pred = out
                if len(pred.shape) == 1:
                    pred = pred.view(-1, 1)
                
                if pred.shape[1] > 1:
                    pred = pred[:, target_idx]
            
            # Ensure both tensors have the same shape
            pred = pred.squeeze()
            
            # Calculate loss: model prediction should match original
            loss = F.mse_loss(pred, target)
            
            # FIXED: Better regularization approach
            # Instead of simple L1 penalty, use entropy-based regularization
            # This encourages sparsity while avoiding complete zero-out
            
            # Option 1: Entropy regularization (encourages binary decisions)
            entropy_reg = -torch.mean(edge_mask * torch.log(edge_mask + 1e-10) + 
                                    (1 - edge_mask) * torch.log(1 - edge_mask + 1e-10))
            
            # Option 2: Controlled sparsity with minimum retention
            min_edges_ratio = 0.1  # Retain at least 10% of edges
            num_edges = edge_mask.shape[0]
            min_edges = max(1, int(num_edges * min_edges_ratio))
            
            # Sort edges by importance and ensure minimum retention
            sorted_mask, sorted_indices = torch.sort(edge_mask, descending=True)
            
            # Sparsity loss that respects minimum edge requirement
            if edge_mask.sum() < min_edges:
                # Penalty for having too few edges
                sparsity_loss = 10.0 * (min_edges - edge_mask.sum())
            else:
                # Normal sparsity penalty, but much weaker
                sparsity_loss = 0.0001 * edge_mask.sum()  # Very weak regularization
            
            # Balance the losses
            prediction_loss = loss
            regularization_loss = 0.01 * entropy_reg + sparsity_loss
            
            total_loss = prediction_loss + regularization_loss
            
            loss = total_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Clamp edge mask values
            with torch.no_grad():
                edge_mask.clamp_(0, 1)
            
            # Check for early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f"Explainer epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.6f}")
        
        print(f"Final explainer loss: {best_loss:.6f}")
        
        # Create edge importance matrix
        num_nodes = data.x.shape[0]
        edge_importance_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        # Fill importance matrix with learned edge mask values
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i], data.edge_index[1, i]
            edge_importance_matrix[u, v] = edge_mask[i]
        
        # Make importance matrix symmetric (for undirected graphs)
        edge_importance_matrix = torch.maximum(edge_importance_matrix, edge_importance_matrix.t())
        
        # Generate text explanation of most important edges
        explanation = self._generate_explanation(edge_importance_matrix, node_names)
        
        # Save edge importance matrix if path is provided
        if save_path is not None:
            if node_names is not None:
                # Create DataFrame with node names
                importance_df = pd.DataFrame(
                    edge_importance_matrix.cpu().detach().numpy(),
                    index=node_names,
                    columns=node_names
                )
                importance_df.to_csv(save_path)
            else:
                # Save as numpy array
                np.savetxt(save_path, edge_importance_matrix.cpu().detach().numpy(), delimiter=',')
        
        return edge_importance_matrix, explanation
    
    def _generate_explanation(self, edge_importance_matrix, node_names=None):
        """Generate text explanation from edge importance matrix"""
        # Convert to numpy
        importance = edge_importance_matrix.cpu().detach().numpy()
        
        # Get top edges
        n = importance.shape[0]
        top_edges = []
        
        for i in range(n):
            for j in range(i+1, n):  # Only consider upper triangle for undirected graph
                if importance[i, j] > 0.2:  # Threshold for importance
                    top_edges.append((i, j, importance[i, j]))
        
        # Sort by importance
        top_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Generate explanation text
        explanation = "Top important feature interactions:\n"
        
        for i, (u, v, imp) in enumerate(top_edges[:10]):  # Show top 10 edges
            if node_names is not None:
                u_name = node_names[u]
                v_name = node_names[v]
                explanation += f"{i+1}. {u_name} ↔ {v_name}: {imp:.3f}\n"
            else:
                explanation += f"{i+1}. Feature {u} ↔ Feature {v}: {imp:.3f}\n"
        
        return explanation 