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

print("Evaluating Catalog Coverage...")
user_pos_train = build_user_item_sets(user_ids, item_ids, train_mask)
user_pos_test  = build_user_item_sets(user_ids, item_ids, test_mask)
eval_users = [u for u in user_pos_test if user_pos_test[u] and u in user_pos_train]

user_text = defaultdict(list)
for idx in train_mask.nonzero(as_tuple=True)[0].tolist():
    user_text[user_ids[idx].item()].append(text_feats[idx])

num_items = metadata['num_items']
all_recs = set()

# Evaluate on first 1000 users to get a stable catalog coverage estimate
sample_users = eval_users[:1000]
print(f"Calculating for {len(sample_users)} users...")

for uid in sample_users:
    seen = user_pos_train[uid]
    cands = [i for i in range(num_items) if i not in seen]
    if not cands: continue
    
    nb = len(cands)
    u_t = torch.tensor([uid] * nb, dtype=torch.long)
    i_t = torch.tensor(cands, dtype=torch.long)
    
    u_tf = torch.stack(user_text[uid]).mean(dim=0) if uid in user_text else torch.zeros(text_feats.shape[1])
    tf = u_tf.unsqueeze(0).expand(nb, -1)
    im_f = item_img_feats[i_t]
    
    with torch.no_grad():
        sc, _, _ = model(u_t, i_t, tf, im_f)
        sc = sc.cpu().numpy()
        
    top10 = [cands[j] for j in np.argsort(sc)[::-1][:10]]
    all_recs.update(top10)

coverage = len(all_recs) / num_items
print(f"Unique items recommended: {len(all_recs)} / {num_items}")
print(f"Catalog Coverage (@10, {len(sample_users)} users): {coverage:.2%}")
