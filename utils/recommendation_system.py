import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq

from models.encoders.multimodal_encoders import RecommendationEncoder
from models.gnn.graph_models import BipartiteGraphRecommender
from models.trust.trust_mechanism import TrustMechanism

class RecommendationSystem:
    """Complete recommendation system with trust awareness"""
    
    def __init__(self, encoder: RecommendationEncoder, gnn: BipartiteGraphRecommender,
                 trust_mechanism: Optional[TrustMechanism] = None,
                 num_users: int = 100, num_items: int = 50):
        
        self.encoder = encoder
        self.gnn = gnn
        self.trust_mechanism = trust_mechanism
        self.num_users = num_users
        self.num_items = num_items
        
        # Item metadata (for display purposes)
        self.item_metadata = {}
        self.user_history = defaultdict(list)
        
        # Recommendation cache
        self.recommendation_cache = {}
        
    def set_item_metadata(self, item_metadata: Dict[int, Dict]):
        """Set metadata for items (names, categories, etc.)"""
        self.item_metadata = item_metadata
    
    def update_user_history(self, user_id: int, item_id: int, rating: float, 
                           review_text: str = "", timestamp: int = 0):
        """Update user interaction history"""
        
        self.user_history[user_id].append({
            'item_id': item_id,
            'rating': rating,
            'review_text': review_text,
            'timestamp': timestamp
        })
    
    def get_user_embedding(self, user_id: int, text_features: torch.Tensor, 
                           images: torch.Tensor) -> torch.Tensor:
        """Get embedding for a specific user"""
        
        # Create dummy item for user embedding
        dummy_item_id = torch.tensor([0])
        user_id_tensor = torch.tensor([user_id])
        
        with torch.no_grad():
            final_emb, embeddings_dict = self.encoder(
                user_id_tensor, dummy_item_id, text_features, images
            )
            user_emb = embeddings_dict['user_emb']
        
        return user_emb.squeeze()
    
    def get_item_embedding(self, item_id: int, text_features: torch.Tensor, 
                           images: torch.Tensor) -> torch.Tensor:
        """Get embedding for a specific item"""
        
        # Create dummy user for item embedding
        dummy_user_id = torch.tensor([0])
        item_id_tensor = torch.tensor([item_id])
        
        with torch.no_grad():
            final_emb, embeddings_dict = self.encoder(
                dummy_user_id, item_id_tensor, text_features, images
            )
            item_emb = embeddings_dict['item_emb']
        
        return item_emb.squeeze()
    
    def predict_user_item_score(self, user_id: int, item_id: int, 
                               text_features: torch.Tensor, images: torch.Tensor,
                               trust_score: Optional[float] = None) -> float:
        """Predict preference score for user-item pair"""
        
        user_id_tensor = torch.tensor([user_id])
        item_id_tensor = torch.tensor([item_id])
        
        with torch.no_grad():
            # Get multimodal embeddings
            final_emb, _ = self.encoder(user_id_tensor, item_id_tensor, text_features, images)
            
            # Get GNN prediction
            prediction, _, _ = self.gnn(user_id_tensor, item_id_tensor, final_emb)
            
            score = prediction.item()
        
        # Adjust score based on trust if provided
        if trust_score is not None and self.trust_mechanism is not None:
            # Lower trust scores reduce prediction confidence
            trust_adjustment = 0.5 + 0.5 * trust_score
            score = score * trust_adjustment
        
        return score
    
    def recommend_items_for_user(self, user_id: int, top_k: int = 10,
                                exclude_seen: bool = True,
                                text_features: Optional[torch.Tensor] = None,
                                images: Optional[torch.Tensor] = None,
                                trust_scores: Optional[Dict[int, float]] = None) -> List[Dict]:
        """Generate recommendations for a specific user"""
        
        # Check cache first
        cache_key = f"{user_id}_{top_k}_{exclude_seen}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        
        # Get seen items for this user
        seen_items = set()
        if exclude_seen and user_id in self.user_history:
            seen_items = set(interaction['item_id'] for interaction in self.user_history[user_id])
        
        # Create dummy features if not provided
        if text_features is None:
            text_features = torch.randn(1, 30)  # Default text feature dimension from metadata
        if images is None:
            images = torch.randn(1, 3, 224, 224)  # Default image size
        
        # Score all items
        item_scores = []
        for item_id in range(self.num_items):
            if item_id in seen_items:
                continue
            
            # Get trust score for this item if available
            item_trust = trust_scores.get(item_id, 1.0) if trust_scores else None
            
            # Predict score
            score = self.predict_user_item_score(
                user_id, item_id, text_features, images, item_trust
            )
            
            item_scores.append((item_id, score))
        
        # Sort by score and get top-k
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = item_scores[:top_k]
        
        # Format recommendations
        recommendations = []
        for item_id, score in top_items:
            rec = {
                'item_id': item_id,
                'score': score,
                'rating_prediction': score * 5.0,  # Convert to 1-5 scale
                'trust_score': trust_scores.get(item_id, 1.0) if trust_scores else 1.0,
                'metadata': self.item_metadata.get(item_id, {}),
                'recommendation_reason': self._generate_reason(user_id, item_id, score)
            }
            recommendations.append(rec)
        
        # Cache results
        self.recommendation_cache[cache_key] = recommendations
        
        return recommendations
    
    def _generate_reason(self, user_id: int, item_id: int, score: float) -> str:
        """Generate explanation for recommendation"""
        
        if score > 0.8:
            return "Highly recommended based on your preferences"
        elif score > 0.6:
            return "Recommended based on similar items you liked"
        elif score > 0.4:
            return "You might like this item"
        else:
            return "New item you might want to try"
    
    def get_similar_items(self, item_id: int, top_k: int = 5,
                         text_features: Optional[torch.Tensor] = None,
                         images: Optional[torch.Tensor] = None) -> List[Dict]:
        """Find similar items to a given item"""
        
        # Get embedding for target item
        if text_features is None:
            text_features = torch.randn(1, 30)
        if images is None:
            images = torch.randn(1, 3, 224, 224)
        
        target_embedding = self.get_item_embedding(item_id, text_features, images)
        
        # Calculate similarity with all other items
        similarities = []
        for other_item_id in range(self.num_items):
            if other_item_id == item_id:
                continue
            
            other_embedding = self.get_item_embedding(other_item_id, text_features, images)
            
            # Cosine similarity
            similarity = torch.cosine_similarity(target_embedding, other_embedding, dim=0)
            similarities.append((other_item_id, similarity.item()))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_items = similarities[:top_k]
        
        # Format results
        results = []
        for similar_item_id, similarity in similar_items:
            result = {
                'item_id': similar_item_id,
                'similarity': similarity,
                'metadata': self.item_metadata.get(similar_item_id, {})
            }
            results.append(result)
        
        return results
    
    def evaluate_recommendations(self, test_interactions: List[Dict], 
                                top_k: int = 10) -> Dict:
        """Evaluate recommendation quality"""
        
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        text_features = torch.randn(1, 30)
        images = torch.randn(1, 3, 224, 224)
        
        for interaction in test_interactions:
            user_id = interaction['user_id']
            actual_item = interaction['item_id']
            actual_rating = interaction['rating']
            
            # Get recommendations
            recommendations = self.recommend_items_for_user(
                user_id, top_k, text_features=text_features, images=images
            )
            
            # Check if actual item is in recommendations
            recommended_items = [rec['item_id'] for rec in recommendations]
            
            # Precision@k
            precision = 1.0 if actual_item in recommended_items else 0.0
            precision_scores.append(precision)
            
            # Recall@k (always 1.0 since we're checking against one item)
            recall_scores.append(precision)
            
            # NDCG@k
            ndcg = 0.0
            if actual_item in recommended_items:
                rank = recommended_items.index(actual_item) + 1
                ndcg = 1.0 / np.log2(rank + 1)
            ndcg_scores.append(ndcg)
        
        return {
            'precision_at_k': np.mean(precision_scores),
            'recall_at_k': np.mean(recall_scores),
            'ndcg_at_k': np.mean(ndcg_scores),
            'num_evaluated': len(test_interactions)
        }
    
    def get_trust_aware_recommendations(self, user_id: int, top_k: int = 10,
                                      trust_threshold: float = 0.5) -> List[Dict]:
        """Get recommendations with trust awareness"""
        
        if self.trust_mechanism is None:
            return self.recommend_items_for_user(user_id, top_k)
        
        # Get trust scores for all items
        item_trust_scores = {}
        for item_id in range(self.num_items):
            # Simulate trust scores based on item popularity
            item_trust_scores[item_id] = self.trust_mechanism.get_client_reputation(f"item_{item_id}")
        
        # Get recommendations with trust filtering
        recommendations = self.recommend_items_for_user(
            user_id, top_k * 2, trust_scores=item_trust_scores
        )
        
        # Filter by trust threshold
        trusted_recommendations = [
            rec for rec in recommendations 
            if rec['trust_score'] >= trust_threshold
        ]
        
        # If not enough trusted recommendations, include some lower trust ones
        if len(trusted_recommendations) < top_k:
            additional_needed = top_k - len(trusted_recommendations)
            low_trust_recs = [
                rec for rec in recommendations 
                if rec['trust_score'] < trust_threshold
            ][:additional_needed]
            trusted_recommendations.extend(low_trust_recs)
        
        return trusted_recommendations[:top_k]

class RecommendationAPI:
    """API wrapper for recommendation system"""
    
    def __init__(self, recommendation_system: RecommendationSystem):
        self.rec_system = recommendation_system
    
    def get_recommendations(self, user_id: int, top_k: int = 10, 
                          trust_aware: bool = False) -> Dict:
        """Get recommendations for a user"""
        
        try:
            if trust_aware:
                recommendations = self.rec_system.get_trust_aware_recommendations(
                    user_id, top_k
                )
            else:
                recommendations = self.rec_system.recommend_items_for_user(
                    user_id, top_k
                )
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendations': recommendations,
                'num_recommendations': len(recommendations)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'user_id': user_id
            }
    
    def get_similar_items(self, item_id: int, top_k: int = 5) -> Dict:
        """Get similar items"""
        
        try:
            similar_items = self.rec_system.get_similar_items(item_id, top_k)
            
            return {
                'success': True,
                'item_id': item_id,
                'similar_items': similar_items,
                'num_similar_items': len(similar_items)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'item_id': item_id
            }
    
    def update_user_interaction(self, user_id: int, item_id: int, rating: float,
                              review_text: str = "", timestamp: int = 0) -> Dict:
        """Update user interaction"""
        
        try:
            self.rec_system.update_user_history(
                user_id, item_id, rating, review_text, timestamp
            )
            
            # Clear recommendation cache for this user
            cache_keys_to_remove = [
                key for key in self.rec_system.recommendation_cache.keys()
                if key.startswith(f"{user_id}_")
            ]
            for key in cache_keys_to_remove:
                del self.rec_system.recommendation_cache[key]
            
            return {
                'success': True,
                'message': f'Updated interaction for user {user_id}, item {item_id}'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test function
def test_recommendation_system():
    """Test the recommendation system"""
    
    print("Testing recommendation system...")
    
    # Create dummy models
    from models.encoders.multimodal_encoders import RecommendationEncoder
    from models.gnn.graph_models import BipartiteGraphRecommender
    
    encoder = RecommendationEncoder(100, 50, 1000)
    gnn = BipartiteGraphRecommender(100, 50)
    
    # Create recommendation system
    rec_system = RecommendationSystem(encoder, gnn, num_users=100, num_items=50)
    
    # Set some dummy item metadata
    item_metadata = {
        i: {
            'name': f'Item {i}',
            'category': f'Category {i % 5}',
            'price': 10.0 + i * 2.5
        } for i in range(50)
    }
    rec_system.set_item_metadata(item_metadata)
    
    # Add some user history
    for user_id in range(5):
        for item_id in range(10):
            rating = np.random.uniform(1, 5)
            rec_system.update_user_history(user_id, item_id, rating)
    
    # Test recommendations
    recommendations = rec_system.recommend_items_for_user(0, top_k=5)
    print(f"Recommendations for user 0: {len(recommendations)} items")
    for rec in recommendations[:3]:
        print(f"  Item {rec['item_id']}: Score {rec['score']:.3f}, Trust {rec['trust_score']:.3f}")
    
    # Test similar items
    similar_items = rec_system.get_similar_items(0, top_k=3)
    print(f"Similar items to item 0: {len(similar_items)} items")
    
    # Test API
    api = RecommendationAPI(rec_system)
    api_response = api.get_recommendations(0, top_k=3)
    print(f"API response success: {api_response['success']}")
    
    print("Recommendation system testing completed!")

if __name__ == "__main__":
    test_recommendation_system()
