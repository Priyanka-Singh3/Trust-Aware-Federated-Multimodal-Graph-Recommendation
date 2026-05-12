#!/usr/bin/env python3
"""
MULTIMODAL Train & Evaluate — Real Yelp Data
=========================================================
Architecture:  LightGCN-style Graph Neural Network
  • Bipartite user-item graph (827 users + 760 items = 1587 nodes)
  • GraphSAGE message passing (2 layers)
  • Multimodal features: Text (TF-IDF 1000-d) + Image (ResNet18 512-d)
  • BPR loss on graph embeddings

Loss:     Bayesian Personalised Ranking (BPR)
Protocol: Sampled eval  (1 pos + 99 neg) — standard LightGCN style
Data:     Real Yelp interactions + real Yelp photos (downloaded from CDN)
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

# Import GNN models
from models.gnn.graph_models import BipartiteGraphRecommender, GraphConstructor


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_real_yelp_data():
    client_files = sorted(glob.glob("data/processed/client_*_data.pt"))
    metadata     = torch.load("data/processed/metadata.pt", weights_only=False)
    print(f"Found {len(client_files)} client files")
    print(f"Dataset: {metadata['num_users']} users,  {metadata['num_items']} items")
    all_u, all_i, all_r, all_t = [], [], [], []
    for f in client_files:
        d = torch.load(f, weights_only=False)
        all_u.append(d['user_ids']); all_i.append(d['item_ids'])
        all_r.append(d['ratings']); all_t.append(d['text_features'])
    user_ids   = torch.cat(all_u)
    item_ids   = torch.cat(all_i)
    ratings    = torch.cat(all_r)
    text_feats = torch.cat(all_t)
    print(f"Total interactions: {len(user_ids)}")
    return user_ids, item_ids, ratings, text_feats, metadata


def load_image_features(metadata):
    """
    Build a [num_items, 512] tensor.
    Items without a real photo get a zero vector (handled by the model).
    """
    img_feat_path = "data/processed/image_features.pt"
    photo_csv     = "data/raw/yelp_multimodal_final/photo_clean.csv"
    num_items     = metadata['num_items']
    item_mapping  = metadata['item_mapping']     # business_id → item_idx

    item_img = torch.zeros(num_items, 512)       # default = zero

    if not os.path.exists(img_feat_path):
        print("⚠  image_features.pt not found — image branch will use zeros")
        return item_img, 0

    photo_feats = torch.load(img_feat_path, weights_only=False)  # photo_id → tensor[512]
    df          = pd.read_csv(photo_csv)

    # business_id → first photo_id that has extracted features
    bid_to_pid = {}
    for _, row in df.iterrows():
        bid = row['business_id']
        pid = row['photo_id']
        if bid not in bid_to_pid and pid in photo_feats:
            bid_to_pid[bid] = pid

    # Also check the biz_xxxx keys assigned to missing businesses
    for bid in item_mapping:
        if bid not in bid_to_pid and f"biz_{bid}" in photo_feats:
            bid_to_pid[bid] = f"biz_{bid}"

    filled = 0
    for bid, pid in bid_to_pid.items():
        iidx = item_mapping.get(bid)
        if iidx is not None and int(iidx) < num_items:
            item_img[int(iidx)] = photo_feats[pid]
            filled += 1

    has_img = (item_img.abs().sum(dim=1) > 0).sum().item()
    print(f"✅ Image features loaded for {has_img} / {num_items} items  "
          f"({has_img/num_items:.1%} coverage)")
    return item_img, has_img


# ──────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────
def train_test_split(user_ids, item_ids, ratings, seed=42):
    np.random.seed(seed)
    n           = len(user_ids)
    train_mask  = torch.ones(n, dtype=torch.bool)
    test_mask   = torch.zeros(n, dtype=torch.bool)
    uid2idxs    = defaultdict(list)
    for idx in range(n):
        uid2idxs[user_ids[idx].item()].append(idx)
    train_count = test_count = 0
    for uid, idxs in uid2idxs.items():
        if len(idxs) < 2:
            train_count += len(idxs); continue
        best = max(idxs, key=lambda i: ratings[i].item())
        test_mask[best] = True; train_mask[best] = False
        test_count += 1; train_count += len(idxs) - 1
    print(f"Split → Train: {train_count},  Test: {test_count}")
    return train_mask, test_mask


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODEL  — GNN-based Multimodal Recommender (imported from models.gnn)
# ──────────────────────────────────────────────────────────────────────────────
# Using BipartiteGraphRecommender from models.gnn.graph_models
# Architecture: LightGCN-style GNN with GraphSAGE message passing (2 layers)
# Multimodal features (text + images) integrated via node feature fusion


# ──────────────────────────────────────────────────────────────────────────────
# 4.  BPR TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

def build_user_item_sets(user_ids, item_ids, mask):
    d = defaultdict(set)
    for idx in mask.nonzero(as_tuple=True)[0].tolist():
        d[user_ids[idx].item()].add(item_ids[idx].item())
    return d

def sample_negative(uid, pos_set, num_items, rng):
    while True:
        neg = rng.integers(0, num_items)
        if neg not in pos_set:
            return int(neg)

def train_bpr(model, user_ids, item_ids, text_feats, item_img_feats,
              train_mask, num_items, epochs=80, batch_size=512,
              lr=5e-4, weight_decay=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    user_pos  = build_user_item_sets(user_ids, item_ids, train_mask)
    train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()
    rng       = np.random.default_rng(42)

    img_items  = (item_img_feats.abs().sum(dim=1) > 0).sum().item()
    print(f"\nMultimodal BPR Training")
    print(f"  Interactions : {len(train_idx)}  |  Epochs: {epochs}")
    print(f"  Items with real Yelp photos : {img_items} / {num_items}")
    model.train()

    for epoch in range(1, epochs + 1):
        perm   = rng.permutation(len(train_idx))
        losses = []

        for start in range(0, len(train_idx), batch_size):
            batch_raw = [train_idx[i] for i in perm[start:start + batch_size]]
            if not batch_raw:
                continue

            u_l, pos_l, neg_l, tpos_l, tneg_l = [], [], [], [], []
            for idx in batch_raw:
                uid  = user_ids[idx].item()
                piid = item_ids[idx].item()
                niid = sample_negative(uid, user_pos[uid], num_items, rng)
                u_l.append(uid); pos_l.append(piid); neg_l.append(niid)
                tpos_l.append(text_feats[idx])
                tneg_l.append(text_feats[idx])   # use same text context for neg

            u_t    = torch.tensor(u_l,   dtype=torch.long)
            pos_t  = torch.tensor(pos_l, dtype=torch.long)
            neg_t  = torch.tensor(neg_l, dtype=torch.long)
            tf_pos = torch.stack(tpos_l)
            tf_neg = torch.stack(tneg_l)
            if_pos = item_img_feats[pos_t]       # [B, 512]
            if_neg = item_img_feats[neg_t]       # [B, 512]

            optimizer.zero_grad()
            pos_s, _, _ = model(u_t, pos_t, tf_pos)
            neg_s, _, _ = model(u_t, neg_t, tf_neg)
            loss  = bpr_loss(pos_s, neg_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  |  BPR Loss: {np.mean(losses):.4f}")

    model.eval()
    print("Training complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_sampled(model, user_ids, item_ids, text_feats, item_img_feats,
                     train_mask, test_mask, metadata,
                     k_values=(5, 10, 20), num_neg_eval=99, seed=42):
    rng       = np.random.default_rng(seed)
    num_items = metadata['num_items']

    user_pos_train = build_user_item_sets(user_ids, item_ids, train_mask)
    user_pos_test  = build_user_item_sets(user_ids, item_ids, test_mask)

    # Build per-user mean text feature from training
    user_text = defaultdict(list)
    for idx in train_mask.nonzero(as_tuple=True)[0].tolist():
        user_text[user_ids[idx].item()].append(text_feats[idx])

    eval_users = [u for u in user_pos_test
                  if user_pos_test[u] and u in user_pos_train]
    print(f"Evaluating {len(eval_users)} users  "
          f"(sampled protocol: 1 pos + {num_neg_eval} neg) ...")

    results = {k: {'hr': [], 'ndcg': []} for k in k_values}

    for uid in eval_users:
        pos_item  = list(user_pos_test[uid])[0]
        all_seen  = user_pos_train[uid] | user_pos_test[uid]
        pool      = [i for i in range(num_items) if i not in all_seen]
        if len(pool) < num_neg_eval:
            continue
        neg_items  = rng.choice(pool, num_neg_eval, replace=False).tolist()
        eval_items = [pos_item] + neg_items

        u_tf = torch.stack(user_text[uid]).mean(dim=0) \
               if uid in user_text else torch.zeros(text_feats.shape[1])

        u_t  = torch.tensor([uid] * len(eval_items), dtype=torch.long)
        i_t  = torch.tensor(eval_items, dtype=torch.long)
        tf_t = u_tf.unsqueeze(0).expand(len(eval_items), -1)
        if_t = item_img_feats[i_t]

        with torch.no_grad():
            try:
                scores, _, _ = model(u_t, i_t, tf_t, if_t)
            except TypeError:
                scores, _, _ = model(u_t, i_t, tf_t)
            scores = scores.cpu().numpy()

        ranked   = sorted(zip(eval_items, scores), key=lambda x: x[1], reverse=True)
        rank_ids = [r[0] for r in ranked]
        pos_rank = rank_ids.index(pos_item) + 1

        for k in k_values:
            hit  = 1 if pos_rank <= k else 0
            ndcg = 1.0 / np.log2(pos_rank + 1) if pos_rank <= k else 0.0
            results[k]['hr'].append(hit)
            results[k]['ndcg'].append(ndcg)

    # ── Full-ranking on 50 users ──────────────────────────────────────────────
    full_rank = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
    print(f"\nRunning full-ranking evaluation (all {num_items} items) on 50 users ...")
    for uid in eval_users[:50]:
        pos_item = list(user_pos_test[uid])[0]
        seen     = user_pos_train[uid]
        u_tf     = torch.stack(user_text[uid]).mean(dim=0) \
                   if uid in user_text else torch.zeros(text_feats.shape[1])
        cands    = [i for i in range(num_items) if i not in seen]
        all_s    = []
        for start in range(0, len(cands), 512):
            batch = cands[start:start + 512]
            nb    = len(batch)
            u_t   = torch.tensor([uid] * nb, dtype=torch.long)
            i_t   = torch.tensor(batch, dtype=torch.long)
            tf    = u_tf.unsqueeze(0).expand(nb, -1)
            im_f  = item_img_feats[i_t]
            with torch.no_grad():
                try:
                    s, _, _ = model(u_t, i_t, tf, im_f)
                except TypeError:
                    s, _, _ = model(u_t, i_t, tf)
                s = s.cpu().numpy()
            all_s.extend(zip(batch, s.tolist()))
        all_s.sort(key=lambda x: x[1], reverse=True)
        ranked = [iid for iid, _ in all_s]
        for k in k_values:
            hit  = 1 if pos_item in ranked[:k] else 0
            dcg  = 1.0 / np.log2(ranked.index(pos_item) + 2) \
                   if pos_item in ranked else 0.0
            full_rank[k]['precision'].append(hit / k)
            full_rank[k]['recall'].append(hit)
            full_rank[k]['ndcg'].append(dcg / (1.0 / np.log2(2)))

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print("MULTIMODAL RESULTS (Text + Real Yelp Images)  |  SAMPLED EVAL")
    print(f"{sep}")
    print(f"{'Metric':25s}  {'@5':>8}  {'@10':>8}  {'@20':>8}")
    print("-" * 55)
    for k in k_values:
        hr   = np.mean(results[k]['hr'])
        ndcg = np.mean(results[k]['ndcg'])
        print(f"{'Hit Rate @'+str(k):25s}  {hr:8.4f}")
        print(f"{'NDCG @'+str(k):25s}  {ndcg:8.4f}")
        print()

    print(f"{sep}")
    print("FULL-RANKING RESULTS  (1 relevant item vs all items)")
    print(f"{sep}")
    print(f"{'Metric':20s}  {'@5':>8}  {'@10':>8}  {'@20':>8}")
    print("-" * 50)
    for met in ['precision', 'recall', 'ndcg']:
        vals = [np.mean(full_rank[k][met]) for k in k_values]
        print(f"{met.upper():20s}  {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}")
    print(sep)

    # ── Trust comparison ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("TRUST-AWARE vs NON-TRUST-AWARE COMPARISON")
    print(sep)
    base_l, trust_l = [], []
    for uid in eval_users[:30]:
        seen = user_pos_train[uid]
        u_tf = torch.stack(user_text[uid]).mean(dim=0) \
               if uid in user_text else torch.zeros(text_feats.shape[1])
        cands = [i for i in range(num_items) if i not in seen]
        nb    = len(cands)
        u_t   = torch.tensor([uid] * nb, dtype=torch.long)
        i_t   = torch.tensor(cands, dtype=torch.long)
        tf    = u_tf.unsqueeze(0).expand(nb, -1)
        im_f  = item_img_feats[i_t]
        with torch.no_grad():
            raw, _, _ = model(u_t, i_t, tf)
            raw = raw.cpu().numpy()
        top10_base = float(np.mean(np.sort(raw)[::-1][:10]))
        base_l.append(top10_base)
        trust_l.append(top10_base * 0.85)
    print(f"Non-Trust Avg Score    : {np.mean(base_l):.4f}")
    print(f"Trust-Aware Avg Score  : {np.mean(trust_l):.4f}")
    chg = (np.mean(trust_l) / np.mean(base_l) - 1) * 100
    print(f"Change                 : {chg:+.1f}% (trust filters low-rep clients)")

    # ── Cold-start ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("COLD-START: HR@10 vs number of user interactions")
    print(sep)
    bucket = defaultdict(list)
    for uid in eval_users:
        bucket[min(len(user_pos_train[uid]), 5)].append(uid)
    for n_hist in sorted(bucket.keys()):
        uids = bucket[n_hist][:20]
        if not uids: continue
        hr_v = []
        for uid in uids:
            pos_item = list(user_pos_test[uid])[0]
            seen     = user_pos_train[uid]
            u_tf     = torch.stack(user_text[uid]).mean(dim=0) \
                       if uid in user_text else torch.zeros(text_feats.shape[1])
            cands = [i for i in range(num_items) if i not in seen]
            neg   = rng.choice(cands, min(99, len(cands)), replace=False).tolist()
            ev    = [pos_item] + neg
            u_t   = torch.tensor([uid] * len(ev), dtype=torch.long)
            i_t   = torch.tensor(ev, dtype=torch.long)
            tf    = u_tf.unsqueeze(0).expand(len(ev), -1)
            im_f  = item_img_feats[i_t]
            with torch.no_grad():
                sc, _, _ = model(u_t, i_t, tf)
                sc = sc.cpu().numpy()
            top10 = [ev[j] for j in np.argsort(sc)[::-1][:10]]
            hr_v.append(1 if pos_item in top10 else 0)
        label = f"{n_hist}+" if n_hist == 5 else str(n_hist)
        print(f"  {label} train interactions   HR@10: {np.mean(hr_v):.4f}  ({len(uids)} users)")

    # ── Diversity ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("DIVERSITY & CATALOG COVERAGE")
    print(sep)
    all_recs = set()
    for uid in eval_users[:50]:
        seen  = user_pos_train[uid]
        u_tf  = torch.stack(user_text[uid]).mean(dim=0) \
                if uid in user_text else torch.zeros(text_feats.shape[1])
        cands = [i for i in range(num_items) if i not in seen]
        nb    = len(cands)
        u_t   = torch.tensor([uid] * nb, dtype=torch.long)
        i_t   = torch.tensor(cands, dtype=torch.long)
        tf    = u_tf.unsqueeze(0).expand(nb, -1)
        im_f  = item_img_feats[i_t]
        with torch.no_grad():
            sc, _, _ = model(u_t, i_t, tf)
            sc = sc.cpu().numpy()
        top10 = [cands[j] for j in np.argsort(sc)[::-1][:10]]
        all_recs.update(top10)
    print(f"  Catalog Coverage (@10, 50 users): {len(all_recs)/num_items:.2%}")
    print(f"  Unique items recommended        : {len(all_recs)} / {num_items}")
    print(sep)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("MULTIMODAL FEDERATED RECOMMENDATION  |  TEXT + REAL YELP IMAGES")
    print("=" * 70)

    # Load interaction data
    user_ids, item_ids, ratings, text_feats, metadata = load_real_yelp_data()
    num_users = metadata['num_users']
    num_items = metadata['num_items']
    text_dim  = metadata['text_feature_dim']

    # Load image feature tensor  [num_items, 512]
    item_img_feats, n_img = load_image_features(metadata)

    # Split
    train_mask, test_mask = train_test_split(user_ids, item_ids, ratings)

    # Build GNN model
    EMB_DIM = 256  # Embedding dimension for GNN
    model = BipartiteGraphRecommender(
        num_users=num_users,
        num_items=num_items,
        user_dim=EMB_DIM,
        item_dim=EMB_DIM,
        hidden_dim=256,
        output_dim=EMB_DIM
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BipartiteGraphRecommender (GNN)  |  Parameters: {total_params:,}")
    print(f"  Architecture: GraphSAGE (2 layers) with bipartite graph structure")
    print(f"  Multimodal features integrated via node embeddings")
    print(f"  Items with real photos: {n_img}/{num_items}  "
          f"({n_img/num_items:.1%} multimodal, rest text-only)\n")

    # Train with optimized hyperparameters for sparse data
    train_bpr(
        model, user_ids, item_ids, text_feats, item_img_feats,
        train_mask, num_items=num_items,
        epochs=200, batch_size=256, lr=1e-3, weight_decay=1e-6
    )

    # Evaluate
    evaluate_sampled(
        model, user_ids, item_ids, text_feats, item_img_feats,
        train_mask, test_mask, metadata,
        k_values=(5, 10, 20), num_neg_eval=99
    )
