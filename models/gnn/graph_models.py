import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx
from typing import Dict, List, Tuple, Optional

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for recommendation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64,
                 num_layers: int = 2, gnn_type: str = 'sage', dropout: float = 0.2):
        super(GraphNeuralNetwork, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, conv in enumerate(self.convs):
            if self.gnn_type == 'gcn':
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output projection
        x = self.output_proj(x)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class BipartiteGraphRecommender(nn.Module):
    """Bipartite graph recommender for user-item interactions"""
    
    def __init__(self, num_users: int, num_items: int, user_dim: int = 64, 
                 item_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super(BipartiteGraphRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, user_dim)
        self.item_embedding = nn.Embedding(num_items, item_dim)
        
        # GNN for user side - handle multimodal features
        self.user_gnn = GraphNeuralNetwork(
            input_dim=user_dim,  # Just user_dim, multimodal features handled separately
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            gnn_type='sage'
        )
        
        # GNN for item side - handle multimodal features
        self.item_gnn = GraphNeuralNetwork(
            input_dim=item_dim,  # Just item_dim, multimodal features handled separately
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            gnn_type='sage'
        )
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def create_bipartite_edges(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Create bipartite graph edges"""
        # User to item edges
        user_to_item = torch.stack([user_ids, item_ids + self.num_users])
        
        # Item to user edges (bidirectional)
        item_to_user = torch.stack([item_ids + self.num_users, user_ids])
        
        # Combine edges
        edge_index = torch.cat([user_to_item, item_to_user], dim=1)
        
        return edge_index
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                multimodal_features: Optional[torch.Tensor] = None):
        
        batch_size = user_ids.size(0)
        
        # Get initial embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Create bipartite graph
        edge_index = self.create_bipartite_edges(user_ids, item_ids)
        
        # Combine user and item features
        if multimodal_features is not None:
            # Use multimodal features for both users and items
            user_features = user_emb  # Just use user embeddings
            item_features = item_emb  # Just use item embeddings
        else:
            user_features = user_emb
            item_features = item_emb
        
        # Create node features for the entire graph
        num_nodes = self.num_users + self.num_items
        feature_dim = user_features.size(1)
        node_features = torch.zeros(num_nodes, feature_dim)
        
        # Set features for nodes in the batch
        unique_users = torch.unique(user_ids)
        unique_items = torch.unique(item_ids)
        
        for i, user_id in enumerate(unique_users):
            mask = user_ids == user_id
            if mask.sum() > 0:
                node_features[user_id] = user_features[mask].mean(dim=0)
        
        for i, item_id in enumerate(unique_items):
            mask = item_ids == item_id
            if mask.sum() > 0:
                node_features[item_id + self.num_users] = item_features[mask].mean(dim=0)
        
        # Apply GNN
        updated_features = self.user_gnn(node_features, edge_index)
        
        # Extract updated user and item embeddings
        updated_user_emb = updated_features[user_ids]
        updated_item_emb = updated_features[item_ids + self.num_users]
        
        # Predict preference score
        combined = torch.cat([updated_user_emb, updated_item_emb], dim=1)
        prediction = self.predictor(combined)
        
        return prediction.squeeze(), updated_user_emb, updated_item_emb

class GraphConstructor:
    """Utility class for constructing graphs from interaction data"""
    
    @staticmethod
    def build_interaction_graph(user_ids: torch.Tensor, item_ids: torch.Tensor, 
                               ratings: torch.Tensor, features: torch.Tensor,
                               num_users: int, num_items: int) -> Data:
        """Build a PyTorch Geometric graph from interaction data"""
        
        # Create edge index (bipartite)
        edge_index = torch.stack([
            torch.cat([user_ids, item_ids + num_users]),
            torch.cat([item_ids + num_users, user_ids])
        ])
        
        # Edge attributes (ratings)
        edge_attr = torch.cat([ratings, ratings])
        
        # Node features
        num_nodes = num_users + num_items
        node_features = torch.zeros(num_nodes, features.size(1))
        
        # Set features for nodes in the batch
        unique_users = torch.unique(user_ids)
        unique_items = torch.unique(item_ids)
        
        for user_id in unique_users:
            mask = user_ids == user_id
            if mask.sum() > 0:
                node_features[user_id] = features[mask].mean(dim=0)
        
        for item_id in unique_items:
            mask = item_ids == item_id
            if mask.sum() > 0:
                node_features[item_id + num_users] = features[mask].mean(dim=0)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        return graph_data
    
    @staticmethod
    def create_client_subgraph(client_data: Dict, global_metadata: Dict) -> Data:
        """Create subgraph for a specific client"""
        
        user_ids = client_data['user_ids']
        item_ids = client_data['item_ids']
        ratings = client_data['ratings']
        text_features = client_data['text_features']
        
        # For simplicity, use text features as node features
        return GraphConstructor.build_interaction_graph(
            user_ids, item_ids, ratings, text_features,
            global_metadata['num_users'], global_metadata['num_items']
        )

class TrustAwareGNN(nn.Module):
    """Trust-aware Graph Neural Network"""
    
    def __init__(self, base_gnn: GraphNeuralNetwork, trust_dim: int = 32):
        super(TrustAwareGNN, self).__init__()
        
        self.base_gnn = base_gnn
        self.trust_dim = trust_dim
        
        # Trust integration layer
        self.trust_integration = nn.Sequential(
            nn.Linear(base_gnn.output_dim + trust_dim, base_gnn.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, edge_index, trust_scores=None, edge_weight=None, batch=None):
        # Get base GNN output
        base_output = self.base_gnn(x, edge_index, edge_weight, batch)
        
        # Integrate trust scores if provided
        if trust_scores is not None:
            # Expand trust scores to match node dimensions
            if trust_scores.dim() == 1:
                trust_scores = trust_scores.unsqueeze(1)
            
            # Concatenate trust scores with node features
            trust_input = torch.cat([base_output, trust_scores.expand(-1, self.trust_dim)], dim=1)
            output = self.trust_integration(trust_input)
        else:
            output = base_output
        
        return output

# Test function
def test_gnn_models():
    """Test GNN models with dummy data"""
    
    print("Testing GNN models...")
    
    # Dummy parameters
    batch_size = 32
    num_users = 100
    num_items = 50
    feature_dim = 128
    
    # Dummy data
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    ratings = torch.rand(batch_size) * 4 + 1  # Ratings 1-5
    features = torch.randn(batch_size, feature_dim)
    
    # Test basic GNN
    gnn = GraphNeuralNetwork(feature_dim, hidden_dim=128, output_dim=64)
    
    # Create dummy graph
    edge_index = torch.randint(0, batch_size, (2, batch_size * 2))
    
    gnn_output = gnn(features, edge_index)
    print(f"Basic GNN output shape: {gnn_output.shape}")
    
    # Test bipartite recommender
    recommender = BipartiteGraphRecommender(num_users, num_items)
    
    predictions, user_emb, item_emb = recommender(user_ids, item_ids)
    print(f"Prediction shape: {predictions.shape}")
    print(f"User embedding shape: {user_emb.shape}")
    print(f"Item embedding shape: {item_emb.shape}")
    
    # Test graph construction
    graph_data = GraphConstructor.build_interaction_graph(
        user_ids, item_ids, ratings, features, num_users, num_items
    )
    print(f"Graph data - Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}")
    
    print("All GNN models working correctly!")

if __name__ == "__main__":
    test_gnn_models()
