import torch
import numpy as np
from train_enhanced_gnn import (
    load_real_yelp_data, load_image_features, train_test_split, evaluate_sampled
)
from models.gnn.enhanced_gnn_models import MultimodalAttentionGNN

print("Loading data...")
user_ids, item_ids, ratings, text_feats, metadata = load_real_yelp_data()
item_img_feats, has_img = load_image_features(metadata)
train_mask, test_mask = train_test_split(user_ids, item_ids, ratings)

print("Loading model...")
model = MultimodalAttentionGNN(
    num_users=metadata['num_users'],
    num_items=metadata['num_items'],
    user_dim=256, item_dim=256, hidden_dim=256, output_dim=256,
    num_layers=3, dropout=0.1,
    text_dim=text_feats.shape[1], image_dim=item_img_feats.shape[1]
)
model.load_state_dict(torch.load('/Users/priyankasingh/Documents/BTP-8/best_enhanced_gnn.pt', map_location='cpu'))
model.eval()

print("Evaluating...")
results = evaluate_sampled(
    model, user_ids, item_ids, text_feats, item_img_feats,
    train_mask, test_mask, metadata,
    k_values=(5, 10, 20), num_neg_eval=99, seed=42
)

# Save results
with open('/Users/priyankasingh/Documents/BTP-8/enhanced_gnn_results.txt', 'w') as f:
    f.write("Enhanced GNN Training Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"HR@10: {np.mean(results[10]['hr']):.4f}\n")
    f.write(f"NDCG@10: {np.mean(results[10]['ndcg']):.4f}\n")
    f.write(f"HR@5: {np.mean(results[5]['hr']):.4f}\n")
    f.write(f"NDCG@5: {np.mean(results[5]['ndcg']):.4f}\n")
    f.write(f"HR@20: {np.mean(results[20]['hr']):.4f}\n")
    f.write(f"NDCG@20: {np.mean(results[20]['ndcg']):.4f}\n")
print("Saved enhanced_gnn_results.txt!")
