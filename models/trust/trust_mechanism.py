import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import math
import logging

logger = logging.getLogger(__name__)

class TrustMechanism:
    """Trust mechanism for federated learning clients"""
    
    def __init__(self, history_length: int = 10, trust_decay: float = 0.9,
                 consistency_weight: float = 0.3, performance_weight: float = 0.4,
                 similarity_weight: float = 0.3):
        
        self.history_length = history_length
        self.trust_decay = trust_decay
        self.consistency_weight = consistency_weight
        self.performance_weight = performance_weight
        self.similarity_weight = similarity_weight
        
        # Trust history for each client
        self.trust_history = defaultdict(lambda: deque(maxlen=history_length))
        self.client_reputations = defaultdict(float)
        
        # Performance baselines
        self.global_loss_history = deque(maxlen=history_length)
        self.baseline_loss = float('inf')
        
        # Update similarity tracking
        self.previous_updates = {}
        
        logger.info("Trust mechanism initialized")
    
    def calculate_consistency_score(self, client_id: str, current_loss: float) -> float:
        """Calculate consistency score based on loss variance"""
        
        if len(self.trust_history[client_id]) < 2:
            return 0.5  # Neutral score for new clients
        
        # Get recent losses
        recent_losses = [entry['loss'] for entry in self.trust_history[client_id]]
        recent_losses.append(current_loss)
        
        # Calculate variance (lower variance = higher consistency)
        loss_variance = np.var(recent_losses)
        
        # Convert to consistency score (inverse of variance, normalized)
        consistency_score = 1.0 / (1.0 + loss_variance)
        
        return consistency_score
    
    def calculate_performance_score(self, client_id: str, current_loss: float, 
                                  num_examples: int) -> float:
        """Calculate performance score based on loss improvement"""
        
        # Update global loss baseline
        if len(self.global_loss_history) > 0:
            avg_global_loss = np.mean(self.global_loss_history)
        else:
            avg_global_loss = current_loss
        
        self.global_loss_history.append(current_loss)
        
        # Performance score based on relative performance
        if current_loss < avg_global_loss:
            performance_score = min(1.0, (avg_global_loss - current_loss) / avg_global_loss + 0.5)
        else:
            performance_score = max(0.0, 0.5 - (current_loss - avg_global_loss) / avg_global_loss)
        
        # Adjust for dataset size (larger datasets get slight bonus)
        size_bonus = min(0.1, math.log(num_examples + 1) / 100)
        performance_score = min(1.0, performance_score + size_bonus)
        
        return performance_score
    
    def calculate_similarity_score(self, client_id: str, current_update: List[np.ndarray]) -> float:
        """Calculate similarity score based on update similarity to previous rounds"""
        
        if client_id not in self.previous_updates:
            self.previous_updates[client_id] = current_update
            return 0.5  # Neutral score for first update
        
        previous_update = self.previous_updates[client_id]
        
        # Calculate cosine similarity for each parameter layer
        similarities = []
        for curr_layer, prev_layer in zip(current_update, previous_update):
            # Flatten layers
            curr_flat = curr_layer.flatten()
            prev_flat = prev_layer.flatten()
            
            # Cosine similarity
            dot_product = np.dot(curr_flat, prev_flat)
            norm_curr = np.linalg.norm(curr_flat)
            norm_prev = np.linalg.norm(prev_flat)
            
            if norm_curr > 0 and norm_prev > 0:
                similarity = dot_product / (norm_curr * norm_prev)
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Average similarity across all layers
        avg_similarity = np.mean(similarities)
        
        # Convert to [0,1] range (cosine similarity is [-1,1])
        similarity_score = (avg_similarity + 1.0) / 2.0
        
        # Update previous update
        self.previous_updates[client_id] = current_update
        
        return similarity_score
    
    def calculate_anomaly_score(self, client_id: str, current_loss: float, 
                               current_update: List[np.ndarray]) -> float:
        """Calculate anomaly score based on statistical outliers"""
        
        # Get all client losses from current round (would need to be passed in)
        # For now, use historical data
        if len(self.global_loss_history) < 3:
            return 0.5
        
        # Calculate Z-score for current loss
        mean_loss = np.mean(self.global_loss_history)
        std_loss = np.std(self.global_loss_history)
        
        if std_loss > 0:
            z_score = abs(current_loss - mean_loss) / std_loss
            # Convert to anomaly score (higher Z-score = more anomalous)
            anomaly_score = min(1.0, z_score / 3.0)
        else:
            anomaly_score = 0.5
        
        return 1.0 - anomaly_score  # Invert so higher is better
    
    def calculate_trust_scores(self, client_metadata: List[Dict], round_num: int) -> List[float]:
        """Calculate trust scores for all clients in current round"""
        
        trust_scores = []
        
        for client_meta in client_metadata:
            client_id = client_meta['client_id']
            current_loss = client_meta['loss']
            num_examples = client_meta['num_examples']
            
            # Calculate individual components
            consistency_score = self.calculate_consistency_score(client_id, current_loss)
            performance_score = self.calculate_performance_score(client_id, current_loss, num_examples)
            
            # For similarity, we'd need the actual model updates
            # For now, use a placeholder
            similarity_score = 0.7  # Placeholder
            
            anomaly_score = self.calculate_anomaly_score(client_id, current_loss, [])
            
            # Combine scores with weights
            trust_score = (
                self.consistency_weight * consistency_score +
                self.performance_weight * performance_score +
                self.similarity_weight * similarity_score
            )
            
            # Apply anomaly penalty
            trust_score *= anomaly_score
            
            # Apply temporal decay based on reputation
            if client_id in self.client_reputations:
                reputation = self.client_reputations[client_id]
                trust_score = self.trust_decay * reputation + (1 - self.trust_decay) * trust_score
            
            # Ensure trust score is in [0,1]
            trust_score = max(0.0, min(1.0, trust_score))
            
            trust_scores.append(trust_score)
            
            # Store in history
            self.trust_history[client_id].append({
                'round': round_num,
                'loss': current_loss,
                'trust_score': trust_score,
                'num_examples': num_examples
            })
            
            # Update reputation
            self.client_reputations[client_id] = trust_score
        
        # Normalize trust scores
        if len(trust_scores) > 1:
            total_trust = sum(trust_scores)
            if total_trust > 0:
                trust_scores = [score / total_trust for score in trust_scores]
        
        logger.info(f"Trust scores calculated: {trust_scores}")
        
        return trust_scores
    
    def update_trust_history(self, client_metadata: List[Dict], trust_scores: List[float]):
        """Update trust history after aggregation"""
        
        for client_meta, trust_score in zip(client_metadata, trust_scores):
            client_id = client_meta['client_id']
            
            # Update long-term reputation with exponential moving average
            if client_id in self.client_reputations:
                self.client_reputations[client_id] = (
                    0.8 * self.client_reputations[client_id] + 0.2 * trust_score
                )
            else:
                self.client_reputations[client_id] = trust_score
    
    def get_client_reputation(self, client_id: str) -> float:
        """Get current reputation score for a client"""
        return self.client_reputations.get(client_id, 0.5)
    
    def get_trust_statistics(self) -> Dict:
        """Get trust mechanism statistics"""
        
        stats = {
            'total_clients': len(self.client_reputations),
            'avg_reputation': np.mean(list(self.client_reputations.values())) if self.client_reputations else 0.0,
            'reputation_std': np.std(list(self.client_reputations.values())) if self.client_reputations else 0.0,
            'global_loss_trend': list(self.global_loss_history) if self.global_loss_history else []
        }
        
        return stats

class TrustAwareAggregator:
    """Trust-aware aggregation strategies"""
    
    @staticmethod
    def trust_weighted_average(client_updates: List[List[np.ndarray]], 
                              trust_scores: List[float]) -> List[np.ndarray]:
        """Perform trust-weighted averaging of client updates"""
        
        if not client_updates:
            return []
        
        # Normalize trust scores
        trust_scores = np.array(trust_scores)
        if trust_scores.sum() > 0:
            trust_scores = trust_scores / trust_scores.sum()
        else:
            trust_scores = np.ones(len(trust_scores)) / len(trust_scores)
        
        # Initialize aggregated parameters
        aggregated_params = []
        
        # For each parameter layer
        for layer_idx in range(len(client_updates[0])):
            # Weighted average of client parameters
            layer_params = []
            for client_idx, client_params in enumerate(client_updates):
                weight = trust_scores[client_idx]
                layer_params.append(weight * client_params[layer_idx])
            
            aggregated_params.append(np.sum(layer_params, axis=0))
        
        return aggregated_params
    
    @staticmethod
    def trimmed_mean_aggregation(client_updates: List[List[np.ndarray]], 
                                trust_scores: List[float], trim_ratio: float = 0.2) -> List[np.ndarray]:
        """Perform trimmed mean aggregation based on trust scores"""
        
        if not client_updates:
            return []
        
        # Sort clients by trust scores
        sorted_indices = np.argsort(trust_scores)
        num_clients = len(client_updates)
        
        # Determine trim range
        trim_count = max(1, int(num_clients * trim_ratio))
        keep_indices = sorted_indices[trim_count:num_clients - trim_count]
        
        # Perform weighted average on remaining clients
        trimmed_updates = [client_updates[i] for i in keep_indices]
        trimmed_trust = [trust_scores[i] for i in keep_indices]
        
        return TrustAwareAggregator.trust_weighted_average(trimmed_updates, trimmed_trust)
    
    @staticmethod
    def reputation_based_sampling(client_updates: List[List[np.ndarray]], 
                                 trust_scores: List[float], 
                                 sample_ratio: float = 0.7) -> List[np.ndarray]:
        """Sample clients based on reputation scores"""
        
        if not client_updates:
            return []
        
        # Convert trust scores to probabilities
        trust_scores = np.array(trust_scores)
        if trust_scores.sum() > 0:
            probabilities = trust_scores / trust_scores.sum()
        else:
            probabilities = np.ones(len(trust_scores)) / len(trust_scores)
        
        # Sample clients
        num_samples = max(1, int(len(client_updates) * sample_ratio))
        sampled_indices = np.random.choice(
            len(client_updates), size=num_samples, replace=False, p=probabilities
        )
        
        # Perform weighted average on sampled clients
        sampled_updates = [client_updates[i] for i in sampled_indices]
        sampled_trust = [trust_scores[i] for i in sampled_indices]
        
        return TrustAwareAggregator.trust_weighted_average(sampled_updates, sampled_trust)

# Test function
def test_trust_mechanism():
    """Test the trust mechanism"""
    
    print("Testing trust mechanism...")
    
    # Initialize trust mechanism
    trust_mech = TrustMechanism()
    
    # Simulate client metadata for multiple rounds
    client_metadata = [
        {'client_id': 'client_0', 'loss': 0.5, 'num_examples': 100},
        {'client_id': 'client_1', 'loss': 0.3, 'num_examples': 150},
        {'client_id': 'client_2', 'loss': 0.8, 'num_examples': 80},
        {'client_id': 'client_3', 'loss': 0.4, 'num_examples': 120},
        {'client_id': 'client_4', 'loss': 0.6, 'num_examples': 90}
    ]
    
    # Calculate trust scores for multiple rounds
    for round_num in range(5):
        trust_scores = trust_mech.calculate_trust_scores(client_metadata, round_num)
        print(f"Round {round_num}: Trust scores = {trust_scores}")
        
        # Simulate some loss changes
        for client in client_metadata:
            client['loss'] *= np.random.uniform(0.8, 1.2)
    
    # Get statistics
    stats = trust_mech.get_trust_statistics()
    print(f"Trust statistics: {stats}")
    
    # Test aggregation methods
    dummy_updates = [[np.random.randn(10, 10) for _ in range(3)] for _ in range(5)]
    
    # Trust-weighted average
    trust_agg = TrustAwareAggregator.trust_weighted_average(dummy_updates, trust_scores)
    print(f"Trust-weighted aggregation completed")
    
    # Trimmed mean
    trimmed_agg = TrustAwareAggregator.trimmed_mean_aggregation(dummy_updates, trust_scores)
    print(f"Trimmed mean aggregation completed")
    
    # Reputation-based sampling
    sampled_agg = TrustAwareAggregator.reputation_based_sampling(dummy_updates, trust_scores)
    print(f"Reputation-based sampling completed")
    
    print("Trust mechanism testing completed!")

if __name__ == "__main__":
    test_trust_mechanism()
