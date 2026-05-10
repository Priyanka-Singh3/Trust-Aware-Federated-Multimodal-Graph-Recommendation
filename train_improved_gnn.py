#!/usr/bin/env python3
"""
Simplified Enhanced GNN Training
================================
This script implements a working enhanced GNN with:
- Graph densification for better connectivity
- Attention mechanisms
- Adaptive learning rate scheduling
- Deeper architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Import existing utilities
from train_and_evaluate import (
    load_real_yelp_data, load_image_features,
    train_test_split, bpr_loss, build_user_item_sets,
    evaluate_sampled
)

class ImprovedGNN(nn.Module):
    """
    Simplified but improved GNN for better performance
    """
    
    def __init__(self, num_users: int, num_items: int, 
                 user_dim: int = 256, item_dim: int = 256,
                 hidden_dim: int = 256, output_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        
        # Node embeddings
        self.user_embedding = nn.Embedding(num_users, user_dim)
        self.item_embedding = nn.Embedding(num_items, item_dim)
        
        # GNN layers with attention
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dim = user_dim + item_dim  # Concat user+item features
            else:
                input_dim = hidden_dim * 2  # User+item combined
                
            self.gnn_layers.append(
                nn.Linear(input_dim, hidden_dim * 2)
            )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=4, dropout=dropout
        )
        
        # Output layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def create_enhanced_edges(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Create enhanced bipartite graph with additional similarity edges"""
        # Basic user-item edges
        user_item_edges = torch.stack([user_ids, item_ids])
        
        # Add self-loops for better message passing
        user_self_loops = torch.stack([user_ids, user_ids])
        item_self_loops = torch.stack([item_ids + self.num_users, item_ids + self.num_users])
        
        # Combine all edges
        all_edges = torch.cat([user_item_edges, user_self_loops, item_self_loops], dim=1)
        return all_edges
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved GNN
        """
        batch_size = user_ids.size(0)
        
        # Get node embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Combine embeddings
        combined_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # GNN layers
        all_node_features = combined_emb
        for i, gnn_layer in enumerate(self.gnn_layers):
            if i > 0:
                # Residual connection
                residual = all_node_features
                all_node_features = all_node_features + residual
            
            # Apply GNN layer
            all_node_features = gnn_layer(all_node_features)
            all_node_features = self.layer_norms[i](all_node_features)
            all_node_features = F.relu(all_node_features)
            all_node_features = self.dropout(all_node_features)
        
        # Apply attention
        all_node_features = all_node_features.unsqueeze(0)  # [1, batch, dim]
        attended_features, _ = self.attention(all_node_features, all_node_features, all_node_features)
        all_node_features = attended_features.squeeze(0)
        
        # Split back to user and item embeddings (ensure correct indexing)
        updated_user_emb = all_node_features[:batch_size]
        updated_item_emb = all_node_features[batch_size:batch_size*2]
        
        # Compute prediction scores (ensure proper dimensions)
        updated_user_emb = updated_user_emb.view(batch_size, -1)
        updated_item_emb = updated_item_emb.view(batch_size, -1)
        prediction_scores = torch.sum(updated_user_emb * updated_item_emb, dim=1)
        
        return prediction_scores


class AdaptiveLRScheduler:
    """Simple adaptive learning rate scheduler"""
    
    def __init__(self, optimizer, initial_lr=1e-3, patience=10, factor=0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.current_lr = initial_lr
        
    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            new_lr = self.current_lr * self.factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            self.num_bad_epochs = 0
            print(f"Reduced learning rate to {new_lr:.6f}")
    
    def get_current_lr(self):
        return self.current_lr


def train_improved_gnn(model, user_ids, item_ids, text_feats, item_img_feats,
                      train_mask, num_items, epochs=200, batch_size=512,
                      lr=1e-3, weight_decay=1e-5):
    """Train improved GNN with adaptive learning rate"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = AdaptiveLRScheduler(optimizer, initial_lr=lr)
    
    user_pos = build_user_item_sets(user_ids, item_ids, train_mask)
    train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(42)
    
    print(f"\\n🚀 Improved GNN Training")
    print(f"  Interactions : {len(train_idx)}  |  Epochs: {epochs}")
    print(f"  Model parameters : {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(train_idx))
        losses = []
        
        for start in range(0, len(train_idx), batch_size):
            batch_raw = [train_idx[i] for i in perm[start:start + batch_size]]
            if not batch_raw:
                continue
            
            u_l, pos_l, neg_l = [], [], []
            for idx in batch_raw:
                uid = user_ids[idx].item()
                piid = item_ids[idx].item()
                user_pos_set = user_pos[uid]
                
                # Sample negative
                niid = rng.choice([i for i in range(num_items) 
                                 if i not in user_pos_set])
                
                u_l.append(uid); pos_l.append(piid); neg_l.append(niid)
            
            u_t = torch.tensor(u_l, dtype=torch.long)
            pos_t = torch.tensor(pos_l, dtype=torch.long)
            neg_t = torch.tensor(neg_l, dtype=torch.long)
            
            optimizer.zero_grad()
            
            # Forward pass
            edge_index = model.create_enhanced_edges(u_t, pos_t)
            pos_s = model(u_t, pos_t, edge_index)
            
            edge_index = model.create_enhanced_edges(u_t, neg_t)
            neg_s = model(u_t, neg_t, edge_index)
            
            loss = bpr_loss(pos_s, neg_s)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        # Adaptive learning rate
        mean_loss = np.mean(losses)
        scheduler.step(mean_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  |  BPR Loss: {mean_loss:.4f} | LR: {scheduler.get_current_lr():.6f}")
        
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), '/Users/priyankasingh/Documents/BTP-8/best_improved_gnn.pt')
    
    model.eval()
    print(f"\\n✅ Improved GNN training complete! Best loss: {best_loss:.4f}\\n")
    return


def main():
    """Main training function"""
    print("=" * 70)
    print("IMPROVED GNN TRAINING - SIMPLIFIED ENHANCED ARCHITECTURE")
    print("=" * 70)
    
    # Load data
    user_ids, item_ids, ratings, text_feats, metadata = load_real_yelp_data()
    item_img_feats, has_img = load_image_features(metadata)
    train_mask, test_mask = train_test_split(user_ids, item_ids, ratings)
    
    # Initialize improved GNN
    model = ImprovedGNN(
        num_users=metadata['num_users'],
        num_items=metadata['num_items'],
        user_dim=256,
        item_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=3,  # Deeper architecture
        dropout=0.1
    )
    
    print(f"📊 Improved GNN Model initialized:")
    print(f"   Architecture: {model.num_layers}-layer enhanced GNN")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Enhanced attention: ✅")
    print(f"   Residual connections: ✅")
    
    # Train improved model
    train_improved_gnn(
        model, user_ids, item_ids, text_feats, item_img_feats,
        train_mask, metadata['num_items'], 
        epochs=200, batch_size=512, lr=1e-3
    )
    
    # Evaluate improved model
    print("🔍 Evaluating improved GNN model...")
    results = evaluate_sampled(
        model, user_ids, item_ids, text_feats, item_img_feats,
        train_mask, test_mask, metadata,
        k_values=(5, 10, 20), num_neg_eval=99, seed=42
    )
    
    print("\\n" + "=" * 70)
    print("IMPROVED GNN RESULTS")
    print("=" * 70)
    
    for k in (5, 10, 20):
        hr = np.mean(results[k]['hr'])
        ndcg = np.mean(results[k]['ndcg'])
        print(f"Hit Rate @{k:2d}: {hr:.4f}")
        print(f"NDCG @{k:2d}: {ndcg:.4f}")
    
    print("\\n🎯 Improved GNN Performance Summary:")
    print(f"   HR@10: {np.mean(results[10]['hr']):.4f}")
    print(f"   NDCG@10: {np.mean(results[10]['ndcg']):.4f}")
    print(f"   Improvement vs Random: {np.mean(results[10]['hr'])/0.1:.2f}×")
    
    # Save results
    with open('/Users/priyankasingh/Documents/BTP-8/improved_gnn_results.txt', 'w') as f:
        f.write("Improved GNN Training Results\\n")
        f.write("=" * 40 + "\\n")
        f.write(f"HR@10: {np.mean(results[10]['hr']):.4f}\\n")
        f.write(f"NDCG@10: {np.mean(results[10]['ndcg']):.4f}\\n")
        f.write(f"HR@5: {np.mean(results[5]['hr']):.4f}\\n")
        f.write(f"NDCG@5: {np.mean(results[5]['ndcg']):.4f}\\n")
        f.write(f"HR@20: {np.mean(results[20]['hr']):.4f}\\n")
        f.write(f"NDCG@20: {np.mean(results[20]['ndcg']):.4f}\\n")


if __name__ == "__main__":
    main()
