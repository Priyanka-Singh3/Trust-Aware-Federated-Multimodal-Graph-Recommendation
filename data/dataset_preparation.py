import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import os
import random
from typing import Tuple, Dict, List

class DatasetPreparation:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def create_sample_amazon_data(self, num_users: int = 100, num_items: int = 50, 
                                 num_interactions: int = 500) -> pd.DataFrame:
        """Create sample Amazon-like review data"""
        
        print(f"Creating sample dataset with {num_users} users, {num_items} items, {num_interactions} interactions")
        
        # Sample review texts
        review_templates = [
            "Great product! Really happy with this purchase.",
            "Not what I expected. Disappointed.",
            "Average quality. Could be better.",
            "Excellent value for money. Highly recommend!",
            "Poor quality. Would not buy again.",
            "Good product. Meets expectations.",
            "Amazing! Better than I thought it would be.",
            "Terrible experience. Waste of money.",
            "Decent product. Nothing special.",
            "Outstanding quality! Worth every penny."
        ]
        
        # Generate interactions
        data = []
        for i in range(num_interactions):
            user_id = f"user_{random.randint(1, num_users)}"
            item_id = f"item_{random.randint(1, num_items)}"
            rating = random.randint(1, 5)
            review_text = random.choice(review_templates)
            timestamp = random.randint(1609459200, 1672531199)  # 2021-2022
            
            data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'review_text': review_text,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(data)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id'])
        
        print(f"Created {len(df)} unique interactions")
        return df
    
    def create_dummy_images(self, num_items: int = 50, image_size: Tuple[int, int] = (224, 224)):
        """Create dummy images for items"""
        
        print(f"Creating {num_items} dummy images...")
        
        image_dir = os.path.join(self.raw_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        for item_id in range(1, num_items + 1):
            # Create random colored image
            img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            image_path = os.path.join(image_dir, f"item_{item_id}.jpg")
            img.save(image_path)
        
        print(f"Dummy images saved to {image_dir}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Preprocess data for federated learning"""
        
        print("Preprocessing data...")
        
        # Encode users and items
        df['user_encoded'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_encoded'] = self.item_encoder.fit_transform(df['item_id'])
        
        # Vectorize review texts
        text_features = self.text_vectorizer.fit_transform(df['review_text']).toarray()
        
        # Create tensors
        user_ids = torch.tensor(df['user_encoded'].values, dtype=torch.long)
        item_ids = torch.tensor(df['item_encoded'].values, dtype=torch.long)
        ratings = torch.tensor(df['rating'].values, dtype=torch.float)
        text_features = torch.tensor(text_features, dtype=torch.float)
        
        # Create metadata
        metadata = {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_),
            'text_feature_dim': text_features.shape[1],
            'user_mapping': dict(zip(self.user_encoder.classes_, self.user_encoder.transform(self.user_encoder.classes_))),
            'item_mapping': dict(zip(self.item_encoder.classes_, self.item_encoder.transform(self.item_encoder.classes_)))
        }
        
        print(f"Preprocessed data: {metadata['num_users']} users, {metadata['num_items']} items")
        
        return user_ids, item_ids, ratings, text_features, metadata
    
    def split_for_federated_learning(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                                   ratings: torch.Tensor, text_features: torch.Tensor,
                                   num_clients: int = 5) -> List[Dict]:
        """Split data for federated learning clients"""
        
        print(f"Splitting data for {num_clients} clients...")
        
        # Get unique users
        unique_users = torch.unique(user_ids)
        
        # Randomly assign users to clients
        np.random.shuffle(unique_users.numpy())
        users_per_client = len(unique_users) // num_clients
        
        client_data = []
        
        for client_id in range(num_clients):
            start_idx = client_id * users_per_client
            if client_id == num_clients - 1:
                end_idx = len(unique_users)
            else:
                end_idx = (client_id + 1) * users_per_client
            
            client_users = unique_users[start_idx:end_idx]
            
            # Get data for this client's users
            mask = torch.isin(user_ids, client_users)
            
            client_dict = {
                'client_id': client_id,
                'user_ids': user_ids[mask],
                'item_ids': item_ids[mask],
                'ratings': ratings[mask],
                'text_features': text_features[mask],
                'users': client_users
            }
            
            client_data.append(client_dict)
            print(f"Client {client_id}: {len(client_users)} users, {mask.sum().item()} interactions")
        
        return client_data
    
    def save_processed_data(self, client_data: List[Dict], metadata: Dict):
        """Save processed data for federated learning"""
        
        print("Saving processed data...")
        
        # Save metadata
        torch.save(metadata, os.path.join(self.processed_dir, "metadata.pt"))
        
        # Save client data
        for client_dict in client_data:
            client_id = client_dict['client_id']
            client_file = os.path.join(self.processed_dir, f"client_{client_id}_data.pt")
            torch.save(client_dict, client_file)
        
        print(f"Data saved to {self.processed_dir}")
    
    def prepare_full_dataset(self, num_users: int = 100, num_items: int = 50, 
                           num_interactions: int = 500, num_clients: int = 5):
        """Full pipeline for dataset preparation"""
        
        # Create sample data
        df = self.create_sample_amazon_data(num_users, num_items, num_interactions)
        
        # Create dummy images
        self.create_dummy_images(num_items)
        
        # Preprocess data
        user_ids, item_ids, ratings, text_features, metadata = self.preprocess_data(df)
        
        # Split for federated learning
        client_data = self.split_for_federated_learning(
            user_ids, item_ids, ratings, text_features, num_clients
        )
        
        # Save processed data
        self.save_processed_data(client_data, metadata)
        
        return client_data, metadata

if __name__ == "__main__":
    # Example usage
    dataset_prep = DatasetPreparation()
    client_data, metadata = dataset_prep.prepare_full_dataset(
        num_users=100,
        num_items=50,
        num_interactions=500,
        num_clients=5
    )
    
    print("Dataset preparation completed!")
    print(f"Metadata: {metadata}")
