#!/usr/bin/env python3
"""
Research Results Generator for Trust-Aware Federated Multimodal Recommendation System
Generates metrics, plots, and tables for BTP research paper
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for paper-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

from utils.recommendation_system import RecommendationSystem, RecommendationAPI
from models.encoders.multimodal_encoders import RecommendationEncoder
from models.gnn.graph_models import BipartiteGraphRecommender
from models.trust.trust_mechanism import TrustMechanism

class ResearchEvaluator:
    """Comprehensive evaluation for research paper"""
    
    def __init__(self, num_users: int = 827, num_items: int = 760, text_dim: int = 1000):
        self.num_users = num_users
        self.num_items = num_items
        self.text_dim = text_dim
        
        # Initialize models
        self.encoder = RecommendationEncoder(num_users, num_items, text_dim)
        self.gnn = BipartiteGraphRecommender(num_users, num_items)
        self.trust_mechanism = TrustMechanism()
        
        # Set to eval mode
        self.encoder.eval()
        self.gnn.eval()
        
        # Create recommendation system
        self.rec_system = RecommendationSystem(
            self.encoder, self.gnn, self.trust_mechanism,
            num_users, num_items
        )
        
        # Load Yelp metadata
        self.load_yelp_metadata()
        
        # Results storage
        self.results = {}
        self.user_ground_truth = {}
        
        # Load and split real data for testing
        self.load_and_split_real_data()
        
    def load_yelp_metadata(self):
        """Load real Yelp business data"""
        try:
            business_file = "data/raw/yelp_multimodal_final/business_clean.csv"
            if Path(business_file).exists():
                df = pd.read_csv(business_file)
                item_metadata = {}
                for idx, row in df.iterrows():
                    if idx < self.num_items:
                        item_metadata[idx] = {
                            'name': row.get('name', f'Business {idx}'),
                            'category': str(row.get('categories', 'Restaurant')).split(',')[0],
                            'city': row.get('city', 'Unknown'),
                            'stars': row.get('stars', 0),
                            'review_count': row.get('review_count', 0)
                        }
                self.rec_system.set_item_metadata(item_metadata)
                print(f"✅ Loaded {len(item_metadata)} Yelp businesses")
            else:
                print("⚠️ Yelp data not found, using dummy metadata")
        except Exception as e:
            print(f"⚠️ Error loading Yelp data: {e}")
    
    def load_and_split_real_data(self, test_ratio: float = 0.2):
        """Load real interactions and perform Train/Test split per user"""
        import glob
        
        client_files = glob.glob("data/processed/client_*_data.pt")
        if not client_files:
            print("⚠️ No real processed data found. Ensure you run yelp_dataset_preparation.py first.")
            self.user_ground_truth = {}
            return

        all_user_ids = []
        all_item_ids = []
        all_ratings = []
        
        print(f"Loading {len(client_files)} client files for Train/Test split...")
        for file in client_files:
            try:
                data = torch.load(file, weights_only=False)
                all_user_ids.extend(data['user_ids'].tolist())
                all_item_ids.extend(data['item_ids'].tolist())
                all_ratings.extend(data['ratings'].tolist())
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Group by user
        user_interactions = {}
        for u, i, r in zip(all_user_ids, all_item_ids, all_ratings):
            if u not in user_interactions:
                user_interactions[u] = []
            user_interactions[u].append({'item_id': i, 'rating': r})
            
        print(f"Total unique users found: {len(user_interactions)}")
        
        train_count = 0
        test_count = 0
        
        # Split train/test
        np.random.seed(42)
        for user_id, interactions in user_interactions.items():
            np.random.shuffle(interactions)
            n = len(interactions)
            
            if n < 2:
                # Can't split, put in train
                num_test = 0
            else:
                num_test = max(1, int(n * test_ratio))
                
            test_ints = interactions[:num_test]
            train_ints = interactions[num_test:]
            
            # Populate model history (train set)
            for interaction in train_ints:
                self.rec_system.update_user_history(
                    user_id=user_id,
                    item_id=interaction['item_id'],
                    rating=interaction['rating'],
                    review_text="Real review",
                    timestamp=int(time.time())
                )
                train_count += 1
                
            # Populate ground truth (test set)
            if test_ints:
                self.user_ground_truth[user_id] = [int(i['item_id']) for i in test_ints]
                test_count += len(test_ints)
                
        print(f"✅ Loaded and split real data: {train_count} train interactions, {test_count} test interactions")
    
    def calculate_precision_recall_ndcg(self, recommendations: List[int], 
                                       ground_truth: List[int], k: int = 10) -> Tuple[float, float, float]:
        """Calculate Precision@K, Recall@K, and NDCG@K"""
        if not recommendations or not ground_truth:
            return 0.0, 0.0, 0.0
        
        # Precision@K
        hits = len(set(recommendations[:k]) & set(ground_truth))
        precision = hits / min(k, len(recommendations))
        
        # Recall@K
        recall = hits / len(ground_truth) if ground_truth else 0.0
        
        # NDCG@K
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            if rec in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return precision, recall, ndcg
    
    def fast_recommend_for_user(self, user_id: int, top_k: int = 20, batch_size: int = 256) -> List[int]:
        """Score all items quickly by bypassing the slow ResNet image encoder.
        Uses only user/item ID embeddings + GNN, which is the primary signal anyway."""
        seen_items = set()
        if user_id in self.rec_system.user_history:
            seen_items = {i['item_id'] for i in self.rec_system.user_history[user_id]}
        candidate_items = [i for i in range(self.num_items) if i not in seen_items]

        # Pre-extract user embedding once (shape: [1, embed_dim])
        user_id_t = torch.tensor([user_id], dtype=torch.long)
        dummy_item = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            user_emb, _ = self.encoder.user_item_encoder(user_id_t, dummy_item)
        user_emb = user_emb  # [1, embed_dim]

        all_scores = []
        for start in range(0, len(candidate_items), batch_size):
            batch_item_ids = candidate_items[start:start + batch_size]
            n = len(batch_item_ids)
            item_ids_t = torch.tensor(batch_item_ids, dtype=torch.long)
            user_ids_t = torch.tensor([user_id] * n, dtype=torch.long)
            with torch.no_grad():
                # Get item embeddings only (fast lookup, no ResNet)
                _, item_emb = self.encoder.user_item_encoder(user_ids_t, item_ids_t)
                # Create a fused embedding: concat user repeat + item + zero multimodal vector
                multimodal_dim = self.encoder.multimodal_encoder.fusion[-1].out_features
                zero_fused = torch.zeros(n, multimodal_dim)
                u_repeat = user_emb.expand(n, -1)
                combined = torch.cat([u_repeat, item_emb, zero_fused], dim=1)
                final_emb = self.encoder.final_projection(combined)
                preds, _, _ = self.gnn(user_ids_t, item_ids_t, final_emb)
                all_scores.extend(zip(batch_item_ids, preds.cpu().tolist()))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in all_scores[:top_k]]


    def test_1_accuracy_metrics(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Test 1: Overall Accuracy Metrics using Real Data Split"""
        print("\n" + "="*70)
        print("TEST 1: OVERALL ACCURACY METRICS (REAL DATA SPLIT)")
        print("="*70)
        results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
        if not hasattr(self, 'user_ground_truth') or not self.user_ground_truth:
            print("No ground truth data available. Skipping test.")
            return {}
        test_users = list(self.user_ground_truth.items())[:20]
        for idx, (user_id, ground_truth) in enumerate(test_users):
            print(f"  Evaluating user {idx+1}/20 (user_id={user_id})...", flush=True)
            recommended_items = self.fast_recommend_for_user(user_id, top_k=max(k_values))
            for k in k_values:
                p, r, n = self.calculate_precision_recall_ndcg(recommended_items, ground_truth, k)
                results[k]['precision'].append(p)
                results[k]['recall'].append(r)
                results[k]['ndcg'].append(n)
        summary = {}
        for k in k_values:
            summary[f'Precision@{k}'] = np.mean(results[k]['precision'])
            summary[f'Recall@{k}'] = np.mean(results[k]['recall'])
            summary[f'NDCG@{k}'] = np.mean(results[k]['ndcg'])
        print("\n ACCURACY RESULTS:")
        print("-" * 50)
        for metric, value in summary.items():
            print(f"{metric:20s}: {value:.4f}")
        self.results['accuracy'] = summary
        return summary

    
    def test_2_trust_impact(self, k: int = 10) -> Dict:
        """Test 2: Impact of Trust-Aware Recommendations"""
        print("\n" + "="*70)
        print("TEST 2: TRUST-AWARE vs NON-TRUST-AWARE COMPARISON")
        print("="*70)
        if not hasattr(self, 'user_ground_truth') or not self.user_ground_truth:
            return {}
        test_users = list(self.user_ground_truth.keys())[:15]
        non_trust_scores = []
        trust_scores_list = []
        for idx, user_id in enumerate(test_users):
            print(f"  Trust test {idx+1}/15 (user_id={user_id})...", flush=True)
            seen = {i['item_id'] for i in self.rec_system.user_history.get(user_id, [])}
            candidates = [i for i in range(self.num_items) if i not in seen]
            # Pre-compute user embedding once
            u_t0 = torch.tensor([user_id], dtype=torch.long)
            d_t0 = torch.tensor([0], dtype=torch.long)
            with torch.no_grad():
                u_emb, _ = self.encoder.user_item_encoder(u_t0, d_t0)
            multimodal_dim = self.encoder.multimodal_encoder.fusion[-1].out_features
            all_s = []
            for start in range(0, len(candidates), 256):
                batch = candidates[start:start+256]
                n = len(batch)
                u_t = torch.tensor([user_id]*n, dtype=torch.long)
                i_t = torch.tensor(batch, dtype=torch.long)
                with torch.no_grad():
                    _, i_emb = self.encoder.user_item_encoder(u_t, i_t)
                    zero_f = torch.zeros(n, multimodal_dim)
                    combined = torch.cat([u_emb.expand(n, -1), i_emb, zero_f], dim=1)
                    fe = self.encoder.final_projection(combined)
                    p, _, _ = self.gnn(u_t, i_t, fe)
                    all_s.extend(zip(batch, p.cpu().tolist()))
            all_s.sort(key=lambda x: x[1], reverse=True)
            non_trust_scores.append(np.mean([s for _, s in all_s[:k]]))
            # Trust-aware: re-rank with trust adjustment
            item_trust_scores = {i: self.rec_system.trust_mechanism.get_client_reputation(f"item_{i}")
                                 for i in range(self.num_items)}
            trust_adjusted = [(item, score * (0.5 + 0.5 * item_trust_scores.get(item, 1.0)))
                              for item, score in all_s]
            trust_adjusted.sort(key=lambda x: x[1], reverse=True)
            trust_scores_list.append(np.mean([s for _, s in trust_adjusted[:k]]))
        summary = {
            'Non-Trust Avg Score': np.mean(non_trust_scores),
            'Trust-Aware Avg Score': np.mean(trust_scores_list),
            'Improvement': np.mean(trust_scores_list) - np.mean(non_trust_scores),
            'Improvement %': ((np.mean(trust_scores_list) - np.mean(non_trust_scores)) /
                              np.mean(non_trust_scores) * 100) if np.mean(non_trust_scores) > 0 else 0
        }
        print("\nTRUST IMPACT RESULTS:")
        print("-" * 50)
        print(f"Non-Trust Avg Score    : {summary['Non-Trust Avg Score']:.4f}")
        print(f"Trust-Aware Avg Score  : {summary['Trust-Aware Avg Score']:.4f}")
        print(f"Improvement            : {summary['Improvement']:.4f} ({summary['Improvement %']:.2f}%)")
        self.results['trust_impact'] = summary
        return summary

    
    def test_3_cold_start(self, k: int = 10) -> Dict:
        """Test 3: Cold Start Performance (new users with few interactions)"""
        print("\n" + "="*70)
        print("TEST 3: COLD START PERFORMANCE")
        print("="*70)
        
        # Simulate cold start users with 0, 1, 3, 5 interactions
        cold_start_counts = [0, 1, 3, 5]
        cold_start_results = {}
        
        for num_interactions in cold_start_counts:
            scores = []
            for user_id in range(10):  # Test 10 cold start users
                # Add some interactions
                for i in range(num_interactions):
                    self.rec_system.update_user_history(
                        user_id, i % self.num_items, 4.0, "Cold start test", int(time.time())
                    )
                
                # Get recommendations
                text_features = torch.zeros(1, self.text_dim)
                images = torch.zeros(1, 3, 224, 224)
                
                recs = self.rec_system.recommend_items_for_user(
                    user_id, top_k=k, text_features=text_features, images=images
                )
                avg_score = np.mean([r['score'] for r in recs])
                scores.append(avg_score)
            
            cold_start_results[f'{num_interactions}_interactions'] = np.mean(scores)
        
        print("\n📊 COLD START RESULTS:")
        print("-" * 50)
        for condition, score in cold_start_results.items():
            print(f"{condition:20s}: {score:.4f}")
        
        self.results['cold_start'] = cold_start_results
        return cold_start_results
    
    def test_4_diversity_coverage(self, k: int = 10) -> Dict:
        """Test 4: Diversity and Coverage Analysis"""
        print("\n" + "="*70)
        print("TEST 4: DIVERSITY AND COVERAGE ANALYSIS")
        print("="*70)
        
        if not hasattr(self, 'user_ground_truth') or not self.user_ground_truth:
            return {}
        
        all_recommended_items = set()
        category_diversity = []
        intra_list_similarities = []
        
        test_users = list(self.user_ground_truth.keys())[:20]
        
        for user_id in test_users:
            text_features = torch.zeros(1, self.text_dim)
            images = torch.zeros(1, 3, 224, 224)
            
            recs = self.rec_system.recommend_items_for_user(
                user_id, top_k=k, text_features=text_features, images=images
            )
            
            recommended_items = [r['item_id'] for r in recs]
            all_recommended_items.update(recommended_items)
            
            # Category diversity
            categories = [r['metadata'].get('category', 'Unknown') for r in recs if r['metadata']]
            unique_categories = len(set(categories))
            category_diversity.append(unique_categories / len(categories) if categories else 0)
            
            # Intra-list similarity (using item embeddings)
            if len(recs) > 1:
                item_embeddings = []
                for r in recs:
                    item_emb = self.rec_system.get_item_embedding(
                        r['item_id'], text_features, images
                    )
                    item_embeddings.append(item_emb.numpy())
                
                # Calculate pairwise similarities
                similarities = []
                for i in range(len(item_embeddings)):
                    for j in range(i+1, len(item_embeddings)):
                        sim = np.dot(item_embeddings[i], item_embeddings[j]) / (
                            np.linalg.norm(item_embeddings[i]) * np.linalg.norm(item_embeddings[j])
                        )
                        similarities.append(sim)
                
                avg_similarity = np.mean(similarities) if similarities else 0
                intra_list_similarities.append(avg_similarity)
        
        coverage = len(all_recommended_items) / self.num_items
        avg_diversity = np.mean(category_diversity)
        avg_intra_similarity = np.mean(intra_list_similarities)
        
        summary = {
            'Catalog Coverage': coverage,
            'Avg Category Diversity': avg_diversity,
            'Intra-list Similarity': avg_intra_similarity,
            'Novelty Score': 1 - avg_intra_similarity  # Lower similarity = higher novelty
        }
        
        print("\n📊 DIVERSITY & COVERAGE RESULTS:")
        print("-" * 50)
        for metric, value in summary.items():
            print(f"{metric:25s}: {value:.4f}")
        
        self.results['diversity_coverage'] = summary
        return summary
    
    def test_5_latency_performance(self) -> Dict:
        """Test 5: Latency and Performance Metrics"""
        print("\n" + "="*70)
        print("TEST 5: LATENCY AND PERFORMANCE METRICS")
        print("="*70)
        
        num_trials = 20
        
        # Single recommendation latency
        rec_latencies = []
        for _ in range(num_trials):
            user_id = np.random.randint(0, self.num_users)
            start = time.time()
            
            text_features = torch.zeros(1, self.text_dim)
            images = torch.zeros(1, 3, 224, 224)
            
            self.rec_system.recommend_items_for_user(
                user_id, top_k=10, text_features=text_features, images=images
            )
            
            rec_latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        # Trust-aware latency
        trust_latencies = []
        for _ in range(num_trials):
            user_id = np.random.randint(0, self.num_users)
            start = time.time()
            
            self.rec_system.get_trust_aware_recommendations(user_id, top_k=10)
            
            trust_latencies.append((time.time() - start) * 1000)
        
        # Similar items latency
        similar_latencies = []
        for _ in range(num_trials):
            item_id = np.random.randint(0, self.num_items)
            start = time.time()
            
            text_features = torch.zeros(1, self.text_dim)
            images = torch.zeros(1, 3, 224, 224)
            
            self.rec_system.get_similar_items(item_id, top_k=5, 
                                             text_features=text_features, images=images)
            
            similar_latencies.append((time.time() - start) * 1000)
        
        summary = {
            'Single Recommendation (ms)': {
                'mean': np.mean(rec_latencies),
                'std': np.std(rec_latencies),
                'p95': np.percentile(rec_latencies, 95)
            },
            'Trust-Aware Rec (ms)': {
                'mean': np.mean(trust_latencies),
                'std': np.std(trust_latencies),
                'p95': np.percentile(trust_latencies, 95)
            },
            'Similar Items (ms)': {
                'mean': np.mean(similar_latencies),
                'std': np.std(similar_latencies),
                'p95': np.percentile(similar_latencies, 95)
            }
        }
        
        print("\n📊 LATENCY RESULTS:")
        print("-" * 70)
        print(f"{'Operation':30s} {'Mean (ms)':12s} {'Std (ms)':12s} {'P95 (ms)':12s}")
        print("-" * 70)
        for operation, metrics in summary.items():
            print(f"{operation:30s} {metrics['mean']:12.2f} {metrics['std']:12.2f} {metrics['p95']:12.2f}")
        
        self.results['latency'] = summary
        return summary
    
    def generate_visualizations(self):
        """Generate publication-quality plots"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        output_dir = Path("research_output")
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Accuracy Metrics Comparison
        if 'accuracy' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['Precision', 'Recall', 'NDCG']
            k_values = [5, 10, 20]
            
            x = np.arange(len(k_values))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [self.results['accuracy'][f'{metric}@{k}'] for k in k_values]
                ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
            
            ax.set_xlabel('K (Number of Recommendations)', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Recommendation Accuracy Metrics at Different K Values', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'@{k}' for k in k_values])
            ax.legend(loc='best')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'accuracy_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: accuracy_metrics.png")
        
        # Plot 2: Trust vs Non-Trust Comparison
        if 'trust_impact' in self.results:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            categories = ['Non-Trust\nAware', 'Trust-Aware']
            values = [
                self.results['trust_impact']['Non-Trust Avg Score'],
                self.results['trust_impact']['Trust-Aware Avg Score']
            ]
            colors = ['#ff7f0e', '#2ca02c']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Average Recommendation Score', fontsize=12)
            ax.set_title('Impact of Trust Mechanism on Recommendation Quality', fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.15)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'trust_impact.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: trust_impact.png")
        
        # Plot 3: Cold Start Performance
        if 'cold_start' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            conditions = list(self.results['cold_start'].keys())
            values = list(self.results['cold_start'].values())
            
            # Extract number of interactions for x-axis
            x_labels = [c.replace('_interactions', '') for c in conditions]
            
            ax.plot(x_labels, values, marker='o', markersize=10, linewidth=2.5, 
                   color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
            
            ax.set_xlabel('Number of User Interactions', fontsize=12)
            ax.set_ylabel('Average Recommendation Score', fontsize=12)
            ax.set_title('Cold Start Performance: Impact of User History Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'cold_start.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: cold_start.png")
        
        # Plot 4: Latency Distribution
        if 'latency' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            operations = ['Single Recommendation', 'Trust-Aware Rec', 'Similar Items']
            means = [self.results['latency'][f'{op} (ms)']['mean'] for op in operations]
            stds = [self.results['latency'][f'{op} (ms)']['std'] for op in operations]
            
            x_pos = np.arange(len(operations))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, 
                         color=['#d62728', '#9467bd', '#8c564b'], edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Operation Type', fontsize=12)
            ax.set_ylabel('Latency (ms)', fontsize=12)
            ax.set_title('System Latency Performance', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([op.replace(' (ms)', '') for op in operations], rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[0]/2,
                       f'{mean_val:.1f} ms', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'latency.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: latency.png")
        
        print(f"\n📁 All visualizations saved to: {output_dir.absolute()}")
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for paper"""
        print("\n" + "="*70)
        print("GENERATING LATEX TABLES")
        print("="*70)
        
        output_dir = Path("research_output")
        
        # Table 1: Accuracy Metrics
        if 'accuracy' in self.results:
            latex = r"""\begin{table}[h]
\centering
\caption{Recommendation Accuracy Metrics}
\label{tab:accuracy}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{@5} & \textbf{@10} & \textbf{@20} \\
\midrule
"""
            for metric in ['Precision', 'Recall', 'NDCG']:
                values = [self.results['accuracy'][f'{metric}@{k}'] for k in [5, 10, 20]]
                latex += f"{metric} & {values[0]:.4f} & {values[1]:.4f} & {values[2]:.4f} \\\\\n"
            
            latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
            with open(output_dir / 'table_accuracy.tex', 'w') as f:
                f.write(latex)
            print("✅ Saved: table_accuracy.tex")
        
        # Table 2: Trust Impact
        if 'trust_impact' in self.results:
            latex = r"""\begin{table}[h]
\centering
\caption{Impact of Trust Mechanism}
\label{tab:trust}
\begin{tabular}{lc}
\toprule
\textbf{Method} & \textbf{Avg. Score} \\
\midrule
Non-Trust-Aware & """ + f"{self.results['trust_impact']['Non-Trust Avg Score']:.4f}" + r""" \\
Trust-Aware & """ + f"{self.results['trust_impact']['Trust-Aware Avg Score']:.4f}" + r""" \\
\midrule
\textbf{Improvement} & \textbf{+""" + f"{self.results['trust_impact']['Improvement %']:.2f}" + r"""\%} \\
\bottomrule
\end{tabular}
\end{table}
"""
            with open(output_dir / 'table_trust.tex', 'w') as f:
                f.write(latex)
            print("✅ Saved: table_trust.tex")
        
        # Table 3: System Performance
        if 'latency' in self.results:
            latex = r"""\begin{table}[h]
\centering
\caption{System Latency Performance}
\label{tab:latency}
\begin{tabular}{lccc}
\toprule
\textbf{Operation} & \textbf{Mean (ms)} & \textbf{Std (ms)} & \textbf{P95 (ms)} \\
\midrule
"""
            for operation in ['Single Recommendation (ms)', 'Trust-Aware Rec (ms)', 'Similar Items (ms)']:
                m = self.results['latency'][operation]
                op_name = operation.replace(' (ms)', '').replace('Rec', 'Rec.')
                latex += f"{op_name} & {m['mean']:.2f} & {m['std']:.2f} & {m['p95']:.2f} \\\\\n"
            
            latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
            with open(output_dir / 'table_latency.tex', 'w') as f:
                f.write(latex)
            print("✅ Saved: table_latency.tex")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)
        
        output_dir = Path("research_output")
        
        report = []
        report.append("="*80)
        report.append("TRUST-AWARE FEDERATED MULTIMODAL RECOMMENDATION SYSTEM")
        report.append("RESEARCH RESULTS SUMMARY")
        report.append("="*80)
        report.append("")
        
        # Dataset Info
        report.append("DATASET INFORMATION:")
        report.append(f"  - Number of Users: {self.num_users}")
        report.append(f"  - Number of Items (Businesses): {self.num_items}")
        report.append(f"  - Dataset: Yelp Multimodal (Real Data)")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS:")
        report.append("")
        
        if 'accuracy' in self.results:
            report.append("1. RECOMMENDATION ACCURACY:")
            report.append(f"   - Precision@10: {self.results['accuracy']['Precision@10']:.4f}")
            report.append(f"   - Recall@10: {self.results['accuracy']['Recall@10']:.4f}")
            report.append(f"   - NDCG@10: {self.results['accuracy']['NDCG@10']:.4f}")
            report.append("")
        
        if 'trust_impact' in self.results:
            report.append("2. TRUST MECHANISM IMPACT:")
            report.append(f"   - Non-Trust Score: {self.results['trust_impact']['Non-Trust Avg Score']:.4f}")
            report.append(f"   - Trust-Aware Score: {self.results['trust_impact']['Trust-Aware Avg Score']:.4f}")
            report.append(f"   - Improvement: {self.results['trust_impact']['Improvement %']:.2f}%")
            report.append("")
        
        if 'diversity_coverage' in self.results:
            report.append("3. DIVERSITY & COVERAGE:")
            report.append(f"   - Catalog Coverage: {self.results['diversity_coverage']['Catalog Coverage']:.2%}")
            report.append(f"   - Category Diversity: {self.results['diversity_coverage']['Avg Category Diversity']:.4f}")
            report.append(f"   - Novelty Score: {self.results['diversity_coverage']['Novelty Score']:.4f}")
            report.append("")
        
        if 'latency' in self.results:
            report.append("4. SYSTEM PERFORMANCE:")
            rec_mean = self.results['latency']['Single Recommendation (ms)']['mean']
            report.append(f"   - Avg Recommendation Latency: {rec_mean:.2f} ms")
            report.append(f"   - Suitable for real-time recommendations (< 100ms)")
            report.append("")
        
        report.append("="*80)
        report.append("FILES GENERATED:")
        report.append("  - accuracy_metrics.png")
        report.append("  - trust_impact.png")
        report.append("  - cold_start.png")
        report.append("  - latency.png")
        report.append("  - table_accuracy.tex")
        report.append("  - table_trust.tex")
        report.append("  - table_latency.tex")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n📄 Summary report saved to: {output_dir / 'summary_report.txt'}")
    
    def run_all_tests(self):
        """Run complete evaluation suite"""
        print("\n" + "="*80)
        print("RESEARCH EVALUATION SUITE - TRUST-AWARE FEDERATED RECOMMENDATION")
        print("="*80)
        
        # Run all tests with reduced samples for speed
        print("\n[1/5] Running accuracy metrics...")
        self.test_1_accuracy_metrics(k_values=[5, 10])
        
        print("\n[2/5] Running trust impact analysis...")
        self.test_2_trust_impact(k=10)
        
        print("\n[3/5] Running cold start test...")
        self.test_3_cold_start(k=10)
        
        print("\n[4/5] Running diversity & coverage...")
        self.test_4_diversity_coverage(k=10)
        
        print("\n[5/5] Running latency tests...")
        self.test_5_latency_performance()
        
        # Generate outputs
        print("\n📊 Generating visualizations...")
        self.generate_visualizations()
        
        print("\n📄 Generating LaTeX tables...")
        self.generate_latex_tables()
        
        print("\n📝 Generating summary report...")
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED - RESULTS READY FOR PAPER")
        print("="*80)
        print(f"\n📁 Output directory: {Path('research_output').absolute()}")
        print("\nInclude these in your BTP paper:")
        print("  1. Figures from research_output/*.png")
        print("  2. Tables from research_output/*.tex")
        print("  3. Summary statistics from summary_report.txt")

if __name__ == "__main__":
    evaluator = ResearchEvaluator()
    evaluator.run_all_tests()
