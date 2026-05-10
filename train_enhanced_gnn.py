#!/usr/bin/env python3
"""
Enhanced GNN Training with Improved Architecture
==============================================
This script implements the enhanced GNN model with:
- Graph densification for better connectivity
- Attention mechanisms for multimodal features
- Adaptive learning rate scheduling
- Graph regularization techniques
- Deeper architecture with residual connections
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Import enhanced GNN models
from models.gnn.enhanced_gnn_models import (
    MultimodalAttentionGNN, 
    GraphDensifier,
    AdaptiveLearningRateScheduler,
    GraphRegularizer
)

# Import existing utilities
from train_and_evaluate import (
    load_real_yelp_data, load_image_features,
    train_test_split, bpr_loss, build_user_item_sets,
    evaluate_sampled
)

def train_enhanced_gnn(model, user_ids, item_ids, text_feats, item_img_feats,
                      train_mask, num_items, epochs=200, batch_size=512,
                      lr=1e-3, weight_decay=1e-5):
    """
    Enhanced training with adaptive learning rate and graph regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = AdaptiveLearningRateScheduler(optimizer, initial_lr=lr)
    
    user_pos = build_user_item_sets(user_ids, item_ids, train_mask)
    train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(42)
    
    # Graph densification
    print("🔗 Applying graph densification...")
    interactions = [(user_ids[i].item(), item_ids[i].item()) 
                   for i in train_idx]
    additional_edges = GraphDensifier.similarity_based_edges(
        model.user_embedding.weight, model.item_embedding.weight,
        torch.tensor(interactions),
        threshold=0.6, max_edges_per_node=8
    )
    
    print(f"   Added {additional_edges.shape[1]} additional edges for better connectivity")
    
    # Create full edge index
    base_edges = model.create_bipartite_edges(user_ids[train_idx], item_ids[train_idx])
    if additional_edges.numel() > 0:
        edge_index = torch.cat([base_edges, additional_edges], dim=1)
    else:
        edge_index = base_edges
    
    print(f"   Total edges: {edge_index.shape[1]}")
    
    img_items = (item_img_feats.abs().sum(dim=1) > 0).sum().item()
    print(f"\\n🚀 Enhanced GNN Training")
    print(f"  Interactions : {len(train_idx)}  |  Epochs: {epochs}")
    print(f"  Items with real Yelp photos : {img_items} / {num_items}")
    print(f"  Graph connectivity : {edge_index.shape[1]} edges")
    print(f"  Model parameters : {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(train_idx))
        losses = []
        reg_losses = []
        
        for start in range(0, len(train_idx), batch_size):
            batch_raw = [train_idx[i] for i in perm[start:start + batch_size]]
            if not batch_raw:
                continue
            
            u_l, pos_l, neg_l, tpos_l, tneg_l = [], [], [], [], []
            for idx in batch_raw:
                uid = user_ids[idx].item()
                piid = item_ids[idx].item()
                user_pos_set = user_pos[uid]
                
                # Sample negative with better strategy
                if len(user_pos_set) < num_items - 1:
                    niid = rng.choice([i for i in range(num_items) 
                                     if i not in user_pos_set])
                else:
                    niid = rng.choice([i for i in range(num_items) 
                                     if i not in user_pos_set or i != piid])
                
                u_l.append(uid); pos_l.append(piid); neg_l.append(niid)
                tpos_l.append(text_feats[idx])
                tneg_l.append(text_feats[idx])  # Use same context
            
            u_t = torch.tensor(u_l, dtype=torch.long)
            pos_t = torch.tensor(pos_l, dtype=torch.long)
            neg_t = torch.tensor(neg_l, dtype=torch.long)
            tf_pos = torch.stack(tpos_l)
            tf_neg = torch.stack(tneg_l)
            if_pos = item_img_feats[pos_t]
            if_neg = item_img_feats[neg_t]
            
            optimizer.zero_grad()
            
            # Forward pass with enhanced GNN
            pos_s, _, _ = model(u_t, pos_t, tf_pos, if_pos, edge_index)
            neg_s, _, _ = model(u_t, neg_t, tf_neg, if_neg, edge_index)
            loss = bpr_loss(pos_s, neg_s)
            
            # Add graph regularization
            if epoch > 50:  # Start regularization after initial learning
                reg_loss = GraphRegularizer.total_regularization(
                    torch.cat([model.user_embedding.weight, model.item_embedding.weight]),
                    edge_index, 
                    model.num_users + model.num_items,
                    alpha=0.001, beta=0.0001
                )
                total_loss = loss + reg_loss
                reg_losses.append(reg_loss.item())
            else:
                total_loss = loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        # Adaptive learning rate scheduling
        mean_loss = np.mean(losses)
        scheduler.step(mean_loss)
        
        if epoch % 10 == 0:
            reg_str = f" | Reg: {np.mean(reg_losses):.6f}" if reg_losses else ""
            print(f"  Epoch {epoch:3d}/{epochs}  |  BPR Loss: {mean_loss:.4f}{reg_str} | LR: {scheduler.get_current_lr():.6f}")
        
        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), '/Users/priyankasingh/Documents/BTP-8/best_enhanced_gnn.pt')
    
    model.eval()
    print(f"\\n✅ Enhanced GNN training complete! Best loss: {best_loss:.4f}\\n")
    return edge_index


def main():
    """Main training function"""
    print("=" * 70)
    print("ENHANCED GNN TRAINING - IMPROVED ARCHITECTURE")
    print("=" * 70)
    
    # Load data
    user_ids, item_ids, ratings, text_feats, metadata = load_real_yelp_data()
    item_img_feats, has_img = load_image_features(metadata)
    train_mask, test_mask = train_test_split(user_ids, item_ids, ratings)
    
    # Initialize enhanced GNN
    model = MultimodalAttentionGNN(
        num_users=metadata['num_users'],
        num_items=metadata['num_items'],
        user_dim=256,
        item_dim=256,
        hidden_dim=256,
        output_dim=256,
        num_layers=3,  # Deeper architecture
        dropout=0.1,
        text_dim=text_feats.shape[1],
        image_dim=item_img_feats.shape[1]
    )
    
    print(f"📊 Enhanced GNN Model initialized:")
    print(f"   Architecture: {model.num_layers}-layer attention GNN")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Multimodal attention: ✅")
    print(f"   Residual connections: ✅")
    print(f"   Graph densification: ✅")
    
    # Train enhanced model
    edge_index = train_enhanced_gnn(
        model, user_ids, item_ids, text_feats, item_img_feats,
        train_mask, metadata['num_items'], 
        epochs=200, batch_size=512, lr=1e-3
    )
    
    # Evaluate enhanced model
    print("🔍 Evaluating enhanced GNN model...")
    results = evaluate_sampled(
        model, user_ids, item_ids, text_feats, item_img_feats,
        train_mask, test_mask, metadata,
        k_values=(5, 10, 20), num_neg_eval=99, seed=42
    )
    
    print("\\n" + "=" * 70)
    print("ENHANCED GNN RESULTS")
    print("=" * 70)
    
    for k in (5, 10, 20):
        hr = np.mean(results[k]['hr'])
        ndcg = np.mean(results[k]['ndcg'])
        print(f"Hit Rate @{k:2d}: {hr:.4f}")
        print(f"NDCG @{k:2d}: {ndcg:.4f}")
    
    print("\\n🎯 Enhanced GNN Performance Summary:")
    print(f"   HR@10: {np.mean(results[10]['hr']):.4f}")
    print(f"   NDCG@10: {np.mean(results[10]['ndcg']):.4f}")
    print(f"   Improvement vs Random: {np.mean(results[10]['hr'])/0.1:.2f}×")
    
    # Save results
    with open('/Users/priyankasingh/Documents/BTP-8/enhanced_gnn_results.txt', 'w') as f:
        f.write("Enhanced GNN Training Results\\n")
        f.write("=" * 40 + "\\n")
        f.write(f"HR@10: {np.mean(results[10]['hr']):.4f}\\n")
        f.write(f"NDCG@10: {np.mean(results[10]['ndcg']):.4f}\\n")
        f.write(f"HR@5: {np.mean(results[5]['hr']):.4f}\\n")
        f.write(f"NDCG@5: {np.mean(results[5]['ndcg']):.4f}\\n")
        f.write(f"HR@20: {np.mean(results[20]['hr']):.4f}\\n")
        f.write(f"NDCG@20: {np.mean(results[20]['ndcg']):.4f}\\n")


if __name__ == "__main__":
    main()
