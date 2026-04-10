#!/usr/bin/env python3
"""
Main entry point for Trust-Aware Federated Multimodal Graph Recommendation System
"""

import argparse
import logging
import sys
import os
import torch
import multiprocessing as mp
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataset_preparation import DatasetPreparation
from server.federated_server import start_server
from client.federated_client import start_client
from utils.recommendation_system import RecommendationSystem, RecommendationAPI
from models.encoders.multimodal_encoders import RecommendationEncoder
from models.gnn.graph_models import BipartiteGraphRecommender
from models.trust.trust_mechanism import TrustMechanism

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_dataset(args):
    """Prepare dataset for federated learning"""
    
    logger.info("Preparing dataset...")
    
    dataset_prep = DatasetPreparation()
    client_data, metadata = dataset_prep.prepare_full_dataset(
        num_users=getattr(args, 'num_users', 100),
        num_items=getattr(args, 'num_items', 50),
        num_interactions=getattr(args, 'num_interactions', 500),
        num_clients=getattr(args, 'num_clients', 5)
    )
    
    logger.info(f"Dataset prepared: {metadata['num_users']} users, {metadata['num_items']} items")
    logger.info(f"Data split among {args.num_clients} clients")
    
    return client_data, metadata

def run_server(args):
    """Run the federated learning server"""
    
    logger.info("Starting federated learning server...")
    
    # Load metadata
    metadata_path = "data/processed/metadata.pt"
    if not os.path.exists(metadata_path):
        logger.error("Metadata not found. Please run dataset preparation first.")
        return
    
    metadata = torch.load(metadata_path, weights_only=False)
    
    # Start server
    start_server(
        num_users=metadata['num_users'],
        num_items=metadata['num_items'],
        text_input_dim=metadata['text_feature_dim'],
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        trust_enabled=args.trust_enabled
    )

def run_client(args):
    """Run a federated learning client"""
    
    logger.info(f"Starting client {args.client_id}...")
    
    start_client(
        client_id=args.client_id,
        server_address=args.server_address
    )

def run_simulation(args):
    """Run a complete federated learning simulation"""
    
    logger.info("Starting complete simulation...")
    
    # Step 1: Prepare dataset
    client_data, metadata = prepare_dataset(args)
    
    # Step 2: Start server in background
    server_process = mp.Process(
        target=run_server,
        args=(args,)
    )
    server_process.start()
    
    # Give server time to start
    import time
    time.sleep(3)
    
    # Step 3: Start clients
    client_processes = []
    for client_id in range(args.num_clients):
        client_process = mp.Process(
            target=run_client,
            args=(argparse.Namespace(
                client_id=client_id,
                server_address=args.server_address
            ),)
        )
        client_process.start()
        client_processes.append(client_process)
    
    # Wait for all processes to complete
    for client_process in client_processes:
        client_process.join()
    
    server_process.join()
    
    logger.info("Simulation completed!")

def test_models(args):
    """Test individual model components"""
    
    logger.info("Testing model components...")
    
    # Test encoders
    from models.encoders.multimodal_encoders import test_encoders
    test_encoders()
    
    # Test GNN models
    from models.gnn.graph_models import test_gnn_models
    test_gnn_models()
    
    # Test trust mechanism
    from models.trust.trust_mechanism import test_trust_mechanism
    test_trust_mechanism()
    
    # Test recommendation system
    from utils.recommendation_system import test_recommendation_system
    test_recommendation_system()
    
    logger.info("All model tests completed!")

def run_interactive_demo(args):
    """Run an interactive demo of the recommendation system"""
    
    logger.info("Starting interactive demo...")
    
    # Load or create models
    try:
        metadata = torch.load("data/processed/metadata.pt", weights_only=False)
        encoder = RecommendationEncoder(
            metadata['num_users'],
            metadata['num_items'],
            metadata['text_feature_dim']
        )
        gnn = BipartiteGraphRecommender(
            metadata['num_users'],
            metadata['num_items']
        )
        trust_mechanism = TrustMechanism()
        
        rec_system = RecommendationSystem(
            encoder, gnn, trust_mechanism,
            metadata['num_users'],
            metadata['num_items']
        )
        
        # Create API
        api = RecommendationAPI(rec_system)
        
        print("\n=== Trust-Aware Recommendation System Demo ===")
        print("Available commands:")
        print("  recommend <user_id> <top_k> - Get recommendations for user")
        print("  similar <item_id> <top_k> - Get similar items")
        print("  update <user_id> <item_id> <rating> - Update user interaction")
        print("  quit - Exit demo")
        print("=" * 50)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                
                elif command[0] == "recommend":
                    if len(command) < 3:
                        print("Usage: recommend <user_id> <top_k>")
                        continue
                    
                    user_id = int(command[1])
                    top_k = int(command[2])
                    
                    result = api.get_recommendations(user_id, top_k, trust_aware=True)
                    
                    if result['success']:
                        print(f"\nTop {result['num_recommendations']} recommendations for user {user_id}:")
                        for i, rec in enumerate(result['recommendations'], 1):
                            print(f"  {i}. Item {rec['item_id']} - Score: {rec['score']:.3f}, "
                                  f"Trust: {rec['trust_score']:.3f}")
                            print(f"     {rec['recommendation_reason']}")
                    else:
                        print(f"Error: {result['error']}")
                
                elif command[0] == "similar":
                    if len(command) < 3:
                        print("Usage: similar <item_id> <top_k>")
                        continue
                    
                    item_id = int(command[1])
                    top_k = int(command[2])
                    
                    result = api.get_similar_items(item_id, top_k)
                    
                    if result['success']:
                        print(f"\nTop {result['num_similar_items']} items similar to item {item_id}:")
                        for i, item in enumerate(result['similar_items'], 1):
                            print(f"  {i}. Item {item['item_id']} - Similarity: {item['similarity']:.3f}")
                    else:
                        print(f"Error: {result['error']}")
                
                elif command[0] == "update":
                    if len(command) < 4:
                        print("Usage: update <user_id> <item_id> <rating>")
                        continue
                    
                    user_id = int(command[1])
                    item_id = int(command[2])
                    rating = float(command[3])
                    
                    result = api.update_user_interaction(user_id, item_id, rating)
                    
                    if result['success']:
                        print(f"Updated: {result['message']}")
                    else:
                        print(f"Error: {result['error']}")
                
                else:
                    print("Unknown command. Available: recommend, similar, update, quit")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nDemo completed!")
        
    except FileNotFoundError:
        logger.error("Data files not found. Please run dataset preparation first.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Trust-Aware Federated Multimodal Graph Recommendation System"
    )
    
    # Global arguments
    parser.add_argument("--num-users", type=int, default=100, help="Number of users")
    parser.add_argument("--num-items", type=int, default=50, help="Number of items")
    parser.add_argument("--num-interactions", type=int, default=500, help="Number of interactions")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--server-address", type=str, default="localhost:8080", 
                       help="Server address")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--trust-enabled", action="store_true", default=True,
                       help="Enable trust mechanism")
    
    # Subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dataset preparation
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--num-users", type=int, default=100, help="Number of users")
    prepare_parser.add_argument("--num-items", type=int, default=50, help="Number of items")
    prepare_parser.add_argument("--num-interactions", type=int, default=500, help="Number of interactions")
    prepare_parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    
    # Server
    server_parser = subparsers.add_parser("server", help="Start federated server")
    server_parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated rounds")
    server_parser.add_argument("--trust-enabled", action="store_true", default=True, help="Enable trust mechanism")
    server_parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    
    # Client
    client_parser = subparsers.add_parser("client", help="Start federated client")
    client_parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    
    # Simulation
    simulate_parser = subparsers.add_parser("simulate", help="Run complete simulation")
    simulate_parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    simulate_parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated rounds")
    simulate_parser.add_argument("--trust-enabled", action="store_true", default=True, help="Enable trust mechanism")
    simulate_parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    
    # Testing
    subparsers.add_parser("test", help="Test model components")
    
    # Demo
    subparsers.add_parser("demo", help="Run interactive demo")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_dataset(args)
    elif args.command == "server":
        run_server(args)
    elif args.command == "client":
        run_client(args)
    elif args.command == "simulate":
        run_simulation(args)
    elif args.command == "test":
        test_models(args)
    elif args.command == "demo":
        run_interactive_demo(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
