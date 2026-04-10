import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from collections import defaultdict
import copy
import logging

from models.gnn.graph_models import BipartiteGraphRecommender
from models.encoders.multimodal_encoders import RecommendationEncoder
from models.trust.trust_mechanism import TrustMechanism

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedServer:
    """Federated Learning Server with Trust-Aware Aggregation"""
    
    def __init__(self, num_users: int, num_items: int, text_input_dim: int = 1000,
                 learning_rate: float = 0.01, trust_enabled: bool = True):
        
        self.num_users = num_users
        self.num_items = num_items
        self.text_input_dim = text_input_dim
        self.learning_rate = learning_rate
        self.trust_enabled = trust_enabled
        
        # Initialize global models
        self.global_encoder = RecommendationEncoder(
            num_users, num_items, text_input_dim
        )
        self.global_gnn = BipartiteGraphRecommender(num_users, num_items)
        
        # Trust mechanism
        if trust_enabled:
            self.trust_mechanism = TrustMechanism()
        
        # Client tracking
        self.client_histories = defaultdict(list)
        self.round_num = 0
        
        # Optimizer for global model updates
        self.encoder_optimizer = torch.optim.Adam(
            self.global_encoder.parameters(), lr=learning_rate
        )
        self.gnn_optimizer = torch.optim.Adam(
            self.global_gnn.parameters(), lr=learning_rate
        )
        
        logger.info("Federated server initialized")
    
    def get_model_parameters(self) -> List[np.ndarray]:
        """Get current global model parameters"""
        encoder_params = [val.cpu().numpy() for val in self.global_encoder.state_dict().values()]
        gnn_params = [val.cpu().numpy() for val in self.global_gnn.state_dict().values()]
        return encoder_params + gnn_params
    
    def set_model_parameters(self, parameters: List[np.ndarray]):
        """Set global model parameters"""
        # Split parameters for encoder and GNN
        encoder_state_dict = self.global_encoder.state_dict()
        gnn_state_dict = self.global_gnn.state_dict()
        
        # Calculate split point
        encoder_param_count = sum(p.numel() for p in encoder_state_dict.values())
        
        # Set encoder parameters
        encoder_params = parameters[:encoder_param_count]
        idx = 0
        for key in encoder_state_dict:
            param_shape = encoder_state_dict[key].shape
            param_size = np.prod(param_shape)
            encoder_state_dict[key] = torch.tensor(
                encoder_params[idx:idx + param_size].reshape(param_shape)
            )
            idx += param_size
        
        # Set GNN parameters
        gnn_params = parameters[encoder_param_count:]
        idx = 0
        for key in gnn_state_dict:
            param_shape = gnn_state_dict[key].shape
            param_size = np.prod(param_shape)
            gnn_state_dict[key] = torch.tensor(
                gnn_params[idx:idx + param_size].reshape(param_shape)
            )
            idx += param_size
        
        self.global_encoder.load_state_dict(encoder_state_dict)
        self.global_gnn.load_state_dict(gnn_state_dict)
    
    def evaluate_global_model(self, test_data: Optional[Dict] = None) -> Dict:
        """Evaluate global model performance"""
        if test_data is None:
            return {"loss": float('inf'), "accuracy": 0.0}
        
        self.global_encoder.eval()
        self.global_gnn.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            # Dummy evaluation - in practice, you'd use actual test data
            user_ids = test_data.get('user_ids', torch.randint(0, self.num_users, (100,)))
            item_ids = test_data.get('item_ids', torch.randint(0, self.num_items, (100,)))
            ratings = test_data.get('ratings', torch.rand(100) * 4 + 1)
            text_features = test_data.get('text_features', torch.randn(100, self.text_input_dim))
            images = test_data.get('images', torch.randn(100, 3, 224, 224))
            
            # Get predictions
            final_emb, _ = self.global_encoder(user_ids, item_ids, text_features, images)
            predictions, _, _ = self.global_gnn(user_ids, item_ids, final_emb)
            
            # Calculate loss
            criterion = nn.MSELoss()
            loss = criterion(predictions, ratings / 5.0)  # Normalize ratings to [0,1]
            total_loss = loss.item()
            
            # Calculate accuracy (within 0.5 of actual rating)
            rating_diff = torch.abs(predictions * 5.0 - ratings)
            correct_predictions = (rating_diff < 0.5).sum().item()
            total_predictions = len(ratings)
        
        return {
            "loss": total_loss,
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }

class TrustAwareFedAvg(fl.server.strategy.Strategy):
    """Trust-Aware Federated Averaging Strategy"""
    
    def __init__(self, server: FederatedServer, **kwargs):
        super().__init__(**kwargs)
        self.server = server
        self.current_round = 0
    
    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        """Initialize global parameters"""
        return fl.common.ndarrays_to_parameters(self.server.get_model_parameters())
    
    def configure_fit(self, server_round: int, parameters: fl.common.NDArrays,
                     client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure clients for training"""
        self.current_round = server_round
        
        # Sample clients
        clients = client_manager.sample(num_clients=client_manager.num_available(), min_num_clients=1)
        
        # Create fit instructions
        fit_ins = fl.common.FitIns(parameters, {})
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]) -> Tuple[Optional[fl.common.NDArrays], Dict]:
        """Aggregate client updates with trust weighting"""
        
        if not results:
            return None, {}
        
        logger.info(f"Aggregating {len(results)} client updates for round {server_round}")
        
        # Extract client parameters and metadata
        client_updates = []
        client_metadata = []
        
        for client_proxy, fit_res in results:
            if fit_res.status.code == fl.common.Code.OK:
                parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
                client_updates.append(parameters)
                
                # Extract metadata from metrics
                metrics = fit_res.metrics or {}
                client_metadata.append({
                    'client_id': metrics.get('client_id', 'unknown'),
                    'loss': metrics.get('loss', float('inf')),
                    'num_examples': fit_res.num_examples,
                    'training_time': metrics.get('training_time', 0.0)
                })
        
        if not client_updates:
            return None, {}
        
        # Calculate trust scores
        if self.server.trust_enabled:
            trust_scores = self.server.trust_mechanism.calculate_trust_scores(
                client_metadata, server_round
            )
        else:
            trust_scores = [1.0] * len(client_updates)
        
        # Trust-weighted aggregation
        aggregated_params = self.trust_weighted_aggregate(client_updates, trust_scores)
        
        # Update server models
        self.server.set_model_parameters(aggregated_params)
        
        # Update trust mechanism
        if self.server.trust_enabled:
            self.server.trust_mechanism.update_trust_history(client_metadata, trust_scores)
        
        # Log results
        avg_trust = np.mean(trust_scores)
        logger.info(f"Round {server_round} completed. Average trust score: {avg_trust:.3f}")
        
        return fl.common.ndarrays_to_parameters(aggregated_params), {
            'avg_trust': avg_trust,
            'num_clients': len(client_updates)
        }
    
    def trust_weighted_aggregate(self, client_updates: List[List[np.ndarray]], 
                                trust_scores: List[float]) -> List[np.ndarray]:
        """Perform trust-weighted aggregation of client updates"""
        
        if not client_updates:
            return []
        
        # Normalize trust scores
        trust_scores = np.array(trust_scores)
        trust_scores = trust_scores / (trust_scores.sum() + 1e-8)
        
        # Initialize aggregated parameters
        aggregated = []
        
        # For each parameter layer
        for layer_idx in range(len(client_updates[0])):
            # Weighted average of client parameters
            layer_params = []
            for client_idx, client_params in enumerate(client_updates):
                weight = trust_scores[client_idx]
                layer_params.append(weight * client_params[layer_idx])
            
            aggregated.append(np.sum(layer_params, axis=0))
        
        return aggregated
    
    def configure_evaluate(self, server_round: int, parameters: fl.common.NDArrays,
                          client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure clients for evaluation"""
        # Skip evaluation for simplicity
        return []
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results"""
        return None, {}
    
    def evaluate(self, server_round: int, parameters: fl.common.NDArrays) -> Tuple[Optional[float], Dict]:
        """Evaluate global model parameters"""
        # Simple evaluation - return dummy loss
        return 0.5, {"accuracy": 0.8}

def start_server(num_users: int, num_items: int, text_input_dim: int = 1000,
                server_address: str = "0.0.0.0:8080", num_rounds: int = 10,
                trust_enabled: bool = True):
    """Start the federated learning server"""
    
    logger.info("Starting federated learning server...")
    
    # Initialize server
    server = FederatedServer(
        num_users=num_users,
        num_items=num_items,
        text_input_dim=text_input_dim,
        trust_enabled=trust_enabled
    )
    
    # Create strategy
    strategy = TrustAwareFedAvg(server)
    
    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    # Example usage
    start_server(
        num_users=100,
        num_items=50,
        text_input_dim=1000,
        server_address="0.0.0.0:8080",
        num_rounds=10,
        trust_enabled=True
    )
