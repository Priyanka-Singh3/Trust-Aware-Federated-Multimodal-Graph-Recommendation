import torch
import numpy as np
from collections import defaultdict
from train_enhanced_gnn import (
    load_real_yelp_data, load_image_features, train_test_split, build_user_item_sets
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
rng = np.random.default_rng(42)
num_items = metadata['num_items']
user_pos_train = build_user_item_sets(user_ids, item_ids, train_mask)
user_pos_test  = build_user_item_sets(user_ids, item_ids, test_mask)

user_text = defaultdict(list)
for idx in train_mask.nonzero(as_tuple=True)[0].tolist():
    user_text[user_ids[idx].item()].append(text_feats[idx])

eval_users = [u for u in user_pos_test if user_pos_test[u] and u in user_pos_train]
print(f"Evaluating {len(eval_users)} users (sampled protocol: 1 pos + 99 neg)...")

k_values = (5, 10, 20)
results = {k: {'hr': [], 'ndcg': []} for k in k_values}

for uid in eval_users:
    pos_item  = list(user_pos_test[uid])[0]
    all_seen  = user_pos_train[uid] | user_pos_test[uid]
    pool      = [i for i in range(num_items) if i not in all_seen]
    if len(pool) < 99: continue
    neg_items  = rng.choice(pool, 99, replace=False).tolist()
    eval_items = [pos_item] + neg_items

    u_tf = torch.stack(user_text[uid]).mean(dim=0) if uid in user_text else torch.zeros(text_feats.shape[1])

    u_t  = torch.tensor([uid] * 100, dtype=torch.long)
    i_t  = torch.tensor(eval_items, dtype=torch.long)
    tf_t = u_tf.unsqueeze(0).expand(100, -1)
    if_t = item_img_feats[i_t]

    with torch.no_grad():
        scores, _, _ = model(u_t, i_t, tf_t, if_t)
        scores = scores.cpu().numpy()

    ranked   = sorted(zip(eval_items, scores), key=lambda x: x[1], reverse=True)
    rank_ids = [r[0] for r in ranked]
    pos_rank = rank_ids.index(pos_item) + 1

    for k in k_values:
        hit  = 1 if pos_rank <= k else 0
        ndcg = 1.0 / np.log2(pos_rank + 1) if pos_rank <= k else 0.0
        results[k]['hr'].append(hit)
        results[k]['ndcg'].append(ndcg)

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
for k in k_values:
    print(f"HR@{k}: {np.mean(results[k]['hr']):.4f} | NDCG@{k}: {np.mean(results[k]['ndcg']):.4f}")
