import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional

from models.gnn.graph_models import BipartiteGraphRecommender, GraphConstructor
from models.encoders.multimodal_encoders import RecommendationEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated Learning Client"""
    
    def __init__(self, client_id: int, client_data: Dict, global_metadata: Dict,
                 learning_rate: float = 0.01, batch_size: int = 32, local_epochs: int = 5):
        
        self.client_id = client_id
        self.client_data = client_data
        self.global_metadata = global_metadata
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Initialize models
        self.encoder = RecommendationEncoder(
            num_users=global_metadata['num_users'],
            num_items=global_metadata['num_items'],
            text_input_dim=global_metadata['text_feature_dim']
        )
        
        self.gnn = BipartiteGraphRecommender(
            num_users=global_metadata['num_users'],
            num_items=global_metadata['num_items']
        )
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Prepare data
        self.prepare_data()
        
        logger.info(f"Client {client_id} initialized with {len(self.dataset)} samples")
    
    def prepare_data(self):
        """Prepare training data for this client"""
        
        user_ids = self.client_data['user_ids']
        item_ids = self.client_data['item_ids']
        ratings = self.client_data['ratings']
        text_features = self.client_data['text_features']
        
        # Create dummy images (in practice, these would be real images)
        images = torch.randn(len(user_ids), 3, 224, 224)
        
        # Create dataset
        self.dataset = TensorDataset(user_ids, item_ids, ratings, text_features, images)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def set_model_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from server"""
        
        # Split parameters for encoder and GNN
        encoder_state_dict = self.encoder.state_dict()
        gnn_state_dict = self.gnn.state_dict()
        
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
        
        self.encoder.load_state_dict(encoder_state_dict)
        self.gnn.load_state_dict(gnn_state_dict)
    
    def get_model_parameters(self) -> List[np.ndarray]:
        """Get current model parameters"""
        
        encoder_params = [val.cpu().numpy() for val in self.encoder.state_dict().values()]
        gnn_params = [val.cpu().numpy() for val in self.gnn.state_dict().values()]
        return encoder_params + gnn_params
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        
        self.encoder.train()
        self.gnn.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_user_ids, batch_item_ids, batch_ratings, batch_text_features, batch_images in self.dataloader:
            
            # Zero gradients
            self.encoder_optimizer.zero_grad()
            self.gnn_optimizer.zero_grad()
            
            # Forward pass
            final_emb, _ = self.encoder(batch_user_ids, batch_item_ids, batch_text_features, batch_images)
            predictions, _, _ = self.gnn(batch_user_ids, batch_item_ids, final_emb)
            
            # Calculate loss
            target = batch_ratings / 5.0  # Normalize to [0,1]
            loss = self.criterion(predictions, target)
            
            # Backward pass
            loss.backward()
            self.encoder_optimizer.step()
            self.gnn_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, num_epochs: int) -> Dict:
        """Train model locally"""
        
        start_time = time.time()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            epoch_losses.append(epoch_loss)
            
            if epoch % 2 == 0:
                logger.info(f"Client {self.client_id} - Epoch {epoch}: Loss = {epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        
        return {
            'avg_loss': avg_loss,
            'epoch_losses': epoch_losses,
            'training_time': training_time,
            'num_samples': len(self.dataset)
        }
    
    def evaluate(self) -> Dict:
        """Evaluate model on local data"""
        
        self.encoder.eval()
        self.gnn.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_user_ids, batch_item_ids, batch_ratings, batch_text_features, batch_images in self.dataloader:
                
                # Forward pass
                final_emb, _ = self.encoder(batch_user_ids, batch_item_ids, batch_text_features, batch_images)
                predictions, _, _ = self.gnn(batch_user_ids, batch_item_ids, final_emb)
                
                # Calculate loss
                target = batch_ratings / 5.0
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
                
                # Calculate accuracy (within 0.5 of actual rating)
                rating_diff = torch.abs(predictions * 5.0 - batch_ratings)
                correct_predictions += (rating_diff < 0.5).sum().item()
                total_predictions += len(batch_ratings)
        
        avg_loss = total_loss / len(self.dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

class FlowerClient(fl.client.Client):
    """Flower client implementation"""
    
    def __init__(self, federated_client: FederatedClient):
        super().__init__()
        self.federated_client = federated_client
    
    def get_parameters(self, config: Dict) -> fl.common.NDArrays:
        """Get model parameters"""
        return self.federated_client.get_model_parameters()
    
    def fit(self, parameters: fl.common.NDArrays, config: Dict) -> Tuple[fl.common.NDArrays, int, Dict]:
        """Train model on local data"""
        
        # Set parameters from server
        self.federated_client.set_model_parameters(parameters)
        
        # Train locally
        training_results = self.federated_client.train(self.federated_client.local_epochs)
        
        # Get updated parameters
        updated_parameters = self.federated_client.get_model_parameters()
        
        # Prepare metrics
        metrics = {
            'client_id': self.federated_client.client_id,
            'loss': training_results['avg_loss'],
            'training_time': training_results['training_time'],
            'num_samples': training_results['num_samples']
        }
        
        return updated_parameters, training_results['num_samples'], metrics
    
    def evaluate(self, parameters: fl.common.NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local data"""
        
        # Set parameters from server
        self.federated_client.set_model_parameters(parameters)
        
        # Evaluate
        eval_results = self.federated_client.evaluate()
        
        # Prepare metrics
        metrics = {
            'client_id': self.federated_client.client_id,
            'accuracy': eval_results['accuracy']
        }
        
        return eval_results['loss'], self.federated_client.num_samples, metrics

def start_client(client_id: int, server_address: str = "localhost:8080"):
    """Start a federated learning client"""
    
    logger.info(f"Starting client {client_id}...")
    
    # Load client data
    import torch
    data_path = f"data/processed/client_{client_id}_data.pt"
    metadata_path = "data/processed/metadata.pt"
    
    try:
        client_data = torch.load(data_path, weights_only=False)
        global_metadata = torch.load(metadata_path, weights_only=False)
    except FileNotFoundError:
        logger.error(f"Data files not found. Please run dataset preparation first.")
        return
    
    # Create federated client
    federated_client = FederatedClient(client_id, client_data, global_metadata)
    
    # Create Flower client
    flower_client = FlowerClient(federated_client)
    
    # Start client
    fl.client.start_client(server_address=server_address, client=flower_client)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start federated learning client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    parser.add_argument("--server-address", type=str, default="localhost:8080", 
                       help="Server address")
    
    args = parser.parse_args()
    
    start_client(args.client_id, args.server_address)
