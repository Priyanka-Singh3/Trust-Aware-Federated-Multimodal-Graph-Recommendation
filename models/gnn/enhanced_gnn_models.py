"""
Enhanced GNN Models for Improved Recommendation Performance
=======================================================
This module implements advanced GNN architectures to address:
1. Graph densification for sparse scenarios
2. Attention mechanisms for better feature weighting
3. Multimodal message passing
4. Adaptive learning rate scheduling
5. Deeper architectures with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import numpy as np
from typing import Optional, Tuple, Dict, List


class GraphDensifier:
    """
    Implements graph densification techniques to improve connectivity in sparse scenarios
    """
    
    @staticmethod
    def similarity_based_edges(user_features: torch.Tensor, item_features: torch.Tensor, 
                           user_item_interactions: torch.Tensor, 
                           threshold: float = 0.7, 
                           max_edges_per_node: int = 5) -> torch.Tensor:
        """
        Create additional edges based on feature similarity to improve graph connectivity
        
        Args:
            user_features: [num_users, feature_dim]
            item_features: [num_items, feature_dim] 
            user_item_interactions: [num_interactions, 2] (user_id, item_id)
            threshold: similarity threshold for edge creation
            max_edges_per_node: maximum new edges per node
            
        Returns:
            Additional edge indices [2, num_additional_edges]
        """
        num_users, feat_dim = user_features.shape
        num_items = item_features.shape[0]
        
        # Normalize features
        user_norm = F.normalize(user_features, p=2, dim=1)
        item_norm = F.normalize(item_features, p=2, dim=1)
        
        # Compute user-user similarity
        user_sim = torch.mm(user_norm, user_norm.t())
        
        # Compute item-item similarity  
        item_sim = torch.mm(item_norm, item_norm.t())
        
        # Get existing edges to avoid duplication
        existing_edges = set()
        for u, i in user_item_interactions:
            existing_edges.add((u.item(), i.item()))
        
        additional_edges = []
        
        # Add top-k similar user edges
        for i in range(num_users):
            # Get top similar users (excluding self and existing connections)
            sim_scores = user_sim[i]
            sim_scores[i] = -1  # Exclude self
            
            # Filter existing connections
            for j in range(num_users):
                if (i, j) in existing_edges or (j, i) in existing_edges:
                    sim_scores[j] = -1
            
            # Get top similar users
            top_k = torch.topk(sim_scores, min(max_edges_per_node, (sim_scores > threshold).sum().item()))
            
            for j in top_k.indices:
                if sim_scores[j] > threshold:
                    additional_edges.append([i, j.item()])
        
        # Add top-k similar item edges
        for i in range(num_items):
            sim_scores = item_sim[i]
            sim_scores[i] = -1  # Exclude self
            
            # Filter existing connections
            for j in range(num_items):
                if (i + num_users, j + num_users) in existing_edges or \
                   (j + num_users, i + num_users) in existing_edges:
                    sim_scores[j] = -1
            
            # Get top similar items
            top_k = torch.topk(sim_scores, min(max_edges_per_node, (sim_scores > threshold).sum().item()))
            
            for j in top_k.indices:
                if sim_scores[j] > threshold:
                    additional_edges.append([i + num_users, j.item() + num_users])
        
        if additional_edges:
            return torch.tensor(additional_edges).t()
        else:
            return torch.empty((2, 0), dtype=torch.long)


class MultimodalAttentionGNN(nn.Module):
    """
    Enhanced GNN with attention mechanisms and multimodal message passing
    """
    
    def __init__(self, num_users: int, num_items: int, 
                 user_dim: int = 256, item_dim: int = 256,
                 hidden_dim: int = 256, output_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1,
                 text_dim: int = 1000, image_dim: int = 512):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Multimodal feature encoders
        self.user_text_proj = nn.Linear(text_dim, hidden_dim // 2)
        self.user_image_proj = nn.Linear(image_dim, hidden_dim // 2)
        self.item_text_proj = nn.Linear(text_dim, hidden_dim // 2)
        self.item_image_proj = nn.Linear(image_dim, hidden_dim // 2)
        
        # Node embeddings
        self.user_embedding = nn.Embedding(num_users, user_dim)
        self.item_embedding = nn.Embedding(num_items, item_dim)
        
        # Multimodal attention layers
        self.user_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout
        )
        self.item_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout
        )
        
        # GNN layers with attention
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dim = hidden_dim * 2  # embedding + multimodal features
            else:
                input_dim = hidden_dim
                
            # Use GAT for attention-based message passing
            self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
        
        # Residual connections
        self.residual_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def create_bipartite_edges(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Create bipartite graph edges with self-loops"""
        # Create user-item edges
        user_item_edges = torch.stack([user_ids, item_ids])
        
        # Add self-loops for message passing
        user_self_loops = torch.stack([user_ids, user_ids])
        item_self_loops = torch.stack([item_ids + self.num_users, item_ids + self.num_users])
        
        # Combine all edges
        all_edges = torch.cat([user_item_edges, user_self_loops, item_self_loops], dim=1)
        return all_edges
    
    def multimodal_feature_fusion(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                               text_features: torch.Tensor, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced multimodal fusion with attention mechanisms
        """
        # Get multimodal features for users and items
        # Handle text features carefully - ensure indices are within bounds
        user_text_feat = self.user_text_proj(text_features[user_ids % text_features.shape[0]])
        user_image_feat = self.user_image_proj(image_features[user_ids % image_features.shape[0]])
        
        item_text_feat = self.item_text_proj(text_features[item_ids % text_features.shape[0]])
        item_image_feat = self.item_image_proj(image_features[item_ids % image_features.shape[0]])
        
        # Concatenate multimodal features
        user_multimodal = torch.cat([user_text_feat, user_image_feat], dim=1)
        item_multimodal = torch.cat([item_text_feat, item_image_feat], dim=1)
        
        # Apply attention to multimodal features
        user_multimodal = user_multimodal.unsqueeze(0)  # [1, batch, dim]
        item_multimodal = item_multimodal.unsqueeze(0)
        
        user_attended, _ = self.user_attention(user_multimodal, user_multimodal, user_multimodal)
        item_attended, _ = self.item_attention(item_multimodal, item_multimodal, item_multimodal)
        
        return user_attended.squeeze(0), item_attended.squeeze(0)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                text_features: torch.Tensor, image_features: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with enhanced GNN architecture
        
        Args:
            user_ids: [batch_size] user indices
            item_ids: [batch_size] item indices  
            text_features: [num_nodes, text_dim] text features
            image_features: [num_nodes, image_dim] image features
            edge_index: [2, num_edges] edge indices (optional)
            
        Returns:
            prediction_scores, updated_user_embeddings, updated_item_embeddings
        """
        batch_size = user_ids.size(0)
        
        # Get node embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Get multimodal features with attention
        user_multimodal, item_multimodal = self.multimodal_feature_fusion(
            user_ids, item_ids, text_features, image_features
        )
        
        # Combine embeddings with multimodal features
        user_combined = torch.cat([user_emb, user_multimodal], dim=1)
        item_combined = torch.cat([item_emb, item_multimodal], dim=1)
        
        # Create edge index if not provided
        if edge_index is None:
            edge_index = self.create_bipartite_edges(user_ids, item_ids)
        else:
            edge_index = edge_index
        
        # GNN message passing with residual connections
        all_node_features = torch.cat([user_combined, item_combined], dim=0)
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            if i > 0:
                # Residual connection
                residual = self.residual_layers[i-1](all_node_features)
                all_node_features = all_node_features + residual
            
            # Apply GNN layer
            all_node_features = gnn_layer(all_node_features, edge_index)
            all_node_features = self.layer_norms[i](all_node_features)
            all_node_features = F.relu(all_node_features)
            all_node_features = self.dropout(all_node_features)
        
        # Split back to user and item embeddings
        updated_user_emb = all_node_features[:batch_size]
        updated_item_emb = all_node_features[batch_size:batch_size*2]
        
        # Compute prediction scores
        combined_emb = torch.cat([updated_user_emb, updated_item_emb], dim=1)
        prediction_scores = torch.sum(updated_user_emb * updated_item_emb, dim=1)
        
        return prediction_scores, updated_user_emb, updated_item_emb


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduling to avoid early convergence
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 initial_lr: float = 1e-3, 
                 patience: int = 10, 
                 factor: float = 0.5,
                 min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.current_lr = initial_lr
        
    def step(self, current_loss: float):
        """
        Update learning rate based on loss improvement
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            # Reduce learning rate
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            self.num_bad_epochs = 0
            print(f"Reduced learning rate to {new_lr:.6f}")
    
    def get_current_lr(self) -> float:
        return self.current_lr


class GraphRegularizer:
    """
    Graph regularization techniques for better training stability
    """
    
    @staticmethod
    def degree_regularization(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Encourage uniform degree distribution
        """
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        return torch.var(deg)
    
    @staticmethod
    def smoothness_regularization(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encourage smoothness in the graph
        """
        row, col = edge_index
        feature_diff = node_features[row] - node_features[col]
        return torch.mean(torch.sum(feature_diff ** 2, dim=1))
    
    @staticmethod
    def total_regularization(node_features: torch.Tensor, edge_index: torch.Tensor, 
                         num_nodes: int, alpha: float = 0.01, beta: float = 0.001) -> torch.Tensor:
        """
        Combined graph regularization
        """
        deg_reg = GraphRegularizer.degree_regularization(edge_index, num_nodes)
        smooth_reg = GraphRegularizer.smoothness_regularization(node_features, edge_index)
        return alpha * deg_reg + beta * smooth_reg


if __name__ == "__main__":
    # Example usage
    print("Enhanced GNN models loaded successfully!")
    print("Available improvements:")
    print("1. GraphDensifier - similarity-based edge creation")
    print("2. MultimodalAttentionGNN - attention + multimodal message passing")
    print("3. AdaptiveLearningRateScheduler - adaptive LR scheduling")
    print("4. GraphRegularizer - graph regularization techniques")
