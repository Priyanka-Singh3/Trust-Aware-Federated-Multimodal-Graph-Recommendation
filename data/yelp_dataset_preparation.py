#!/usr/bin/env python3
"""
Dataset Preparation for REAL Yelp Multimodal Dataset
Loads actual Yelp data with reviews, businesses, and photos
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import os
import json
from typing import Tuple, Dict, List
from pathlib import Path

class YelpDatasetPreparation:
    """Dataset preparation for real Yelp multimodal data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = Path(data_dir) / "raw" / "yelp_multimodal_final"
        self.processed_dir = Path(data_dir) / "processed"
        self.image_dir = self.raw_dir / "images"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Data storage
        self.df_reviews = None
        self.df_business = None
        self.df_photos = None
        
    def load_yelp_data(self) -> bool:
        """Load real Yelp data from CSV files"""
        
        print("\n" + "="*70)
        print("📦 Loading Real Yelp Multimodal Dataset")
        print("="*70)
        
        # Check if data exists
        if not self.raw_dir.exists():
            print(f"❌ Data directory not found: {self.raw_dir}")
            print("   Run: python data/download_yelp_csvs.py")
            return False
        
        # Load review data
        review_file = self.raw_dir / "review_clean.csv"
        if review_file.exists():
            self.df_reviews = pd.read_csv(review_file)
            print(f"✅ Loaded {len(self.df_reviews)} reviews")
        else:
            print(f"⚠️  Review file not found: {review_file}")
            return False
        
        # Load business data
        business_file = self.raw_dir / "business_clean.csv"
        if business_file.exists():
            self.df_business = pd.read_csv(business_file)
            print(f"✅ Loaded {len(self.df_business)} businesses")
        else:
            print(f"⚠️  Business file not found: {business_file}")
        
        # Load photo data
        photo_file = self.raw_dir / "photo_clean.csv"
        if photo_file.exists():
            self.df_photos = pd.read_csv(photo_file)
            print(f"✅ Loaded {len(self.df_photos)} photo records")
        else:
            print(f"⚠️  Photo file not found: {photo_file}")
        
        print("="*70)
        return True
    
    def prepare_interaction_data(self) -> pd.DataFrame:
        """Prepare user-item interaction data from reviews"""
        
        if self.df_reviews is None:
            raise ValueError("Reviews not loaded. Call load_yelp_data() first.")
        
        print("\n🔄 Preparing interaction data...")
        
        # Create interaction dataframe
        df_interactions = pd.DataFrame({
            'user_id': self.df_reviews['user_id'],
            'item_id': self.df_reviews['business_id'],
            'rating': self.df_reviews['stars'],
            'review_text': self.df_reviews['text'],
            'review_id': self.df_reviews['review_id'],
            'reactions': self.df_reviews.get('reactions', 0)
        })
        
        # Remove duplicates (user-business pairs)
        df_interactions = df_interactions.drop_duplicates(subset=['user_id', 'item_id'])
        
        print(f"✅ Created {len(df_interactions)} unique interactions")
        print(f"   Users: {df_interactions['user_id'].nunique()}")
        print(f"   Items: {df_interactions['item_id'].nunique()}")
        
        return df_interactions
    
    def load_images_for_businesses(self, business_ids: List[str]) -> Dict[str, str]:
        """Load image paths for given businesses"""
        
        image_paths = {}
        
        if not self.image_dir.exists():
            print(f"⚠️  Image directory not found: {self.image_dir}")
            return image_paths
        
        # Map businesses to photos
        if self.df_photos is not None:
            business_photos = self.df_photos[self.df_photos['business_id'].isin(business_ids)]
            
            for _, row in business_photos.iterrows():
                photo_id = row['photo_id']
                business_id = row['business_id']
                
                # Check if image exists
                img_path = self.image_dir / f"{photo_id}.jpg"
                if img_path.exists():
                    if business_id not in image_paths:
                        image_paths[business_id] = str(img_path)
        
        print(f"✅ Found images for {len(image_paths)} businesses")
        return image_paths
    
    def preprocess_for_federated_learning(self, df: pd.DataFrame) -> Tuple:
        """Preprocess data for federated learning"""
        
        print("\n⚙️  Preprocessing for federated learning...")
        
        # Filter users and items with minimum interactions
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        min_user_interactions = 2
        min_item_interactions = 2
        
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index
        
        df_filtered = df[
            df['user_id'].isin(valid_users) & 
            df['item_id'].isin(valid_items)
        ].copy()
        
        print(f"   Filtered: {len(df_filtered)} interactions")
        
        # Encode users and items
        df_filtered['user_encoded'] = self.user_encoder.fit_transform(df_filtered['user_id'])
        df_filtered['item_encoded'] = self.item_encoder.fit_transform(df_filtered['item_id'])
        
        # Vectorize review texts
        print("   Vectorizing review texts...")
        text_features = self.text_vectorizer.fit_transform(df_filtered['review_text'].fillna('')).toarray()
        
        # Create tensors
        user_ids = torch.tensor(df_filtered['user_encoded'].values, dtype=torch.long)
        item_ids = torch.tensor(df_filtered['item_encoded'].values, dtype=torch.long)
        ratings = torch.tensor(df_filtered['rating'].values, dtype=torch.float)
        text_features = torch.tensor(text_features, dtype=torch.float)
        
        # Get business images
        business_ids = df_filtered['item_id'].unique().tolist()
        image_paths = self.load_images_for_businesses(business_ids)
        
        # Create metadata
        metadata = {
            'num_users': len(self.user_encoder.classes_),
            'num_items': len(self.item_encoder.classes_),
            'text_feature_dim': text_features.shape[1],
            'user_mapping': dict(zip(self.user_encoder.classes_, 
                                   self.user_encoder.transform(self.user_encoder.classes_))),
            'item_mapping': dict(zip(self.item_encoder.classes_, 
                                   self.item_encoder.transform(self.item_encoder.classes_))),
            'image_paths': image_paths,
            'num_images': len(image_paths)
        }
        
        print(f"✅ Preprocessed: {metadata['num_users']} users, {metadata['num_items']} items")
        print(f"   Text features: {metadata['text_feature_dim']} dimensions")
        print(f"   Images available: {metadata['num_images']}")
        
        return user_ids, item_ids, ratings, text_features, metadata, df_filtered
    
    def split_for_federated_learning(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                                   ratings: torch.Tensor, text_features: torch.Tensor,
                                   num_clients: int = 5) -> List[Dict]:
        """Split data for federated learning clients by users"""
        
        print(f"\n🔄 Splitting data for {num_clients} clients...")
        
        # Get unique users
        unique_users = torch.unique(user_ids)
        num_users = len(unique_users)
        
        # Shuffle users for random distribution
        np.random.seed(42)
        shuffled_users = unique_users[torch.randperm(num_users)]
        
        # Split users among clients
        users_per_client = num_users // num_clients
        client_data = []
        
        for client_id in range(num_clients):
            # Get user range for this client
            start_idx = client_id * users_per_client
            if client_id == num_clients - 1:
                end_idx = num_users
            else:
                end_idx = (client_id + 1) * users_per_client
            
            client_users = shuffled_users[start_idx:end_idx]
            
            # Get all interactions for these users
            mask = torch.isin(user_ids, client_users)
            
            client_dict = {
                'client_id': client_id,
                'user_ids': user_ids[mask],
                'item_ids': item_ids[mask],
                'ratings': ratings[mask],
                'text_features': text_features[mask],
                'num_users': len(client_users),
                'num_interactions': mask.sum().item()
            }
            
            client_data.append(client_dict)
            print(f"   Client {client_id}: {client_dict['num_users']} users, {client_dict['num_interactions']} interactions")
        
        return client_data
    
    def save_processed_data(self, client_data: List[Dict], metadata: Dict):
        """Save processed data for federated learning"""
        
        print("\n💾 Saving processed data...")
        
        # Save metadata
        metadata_file = self.processed_dir / "metadata.pt"
        torch.save(metadata, metadata_file)
        print(f"✅ Saved metadata: {metadata_file}")
        
        # Save client data
        for client_dict in client_data:
            client_id = client_dict['client_id']
            client_file = self.processed_dir / f"client_{client_id}_data.pt"
            torch.save(client_dict, client_file)
        
        print(f"✅ Saved {len(client_data)} client files to {self.processed_dir}")
    
    def prepare_yelp_for_federated_learning(self, num_clients: int = 5):
        """Full pipeline to prepare Yelp data for federated learning"""
        
        print("\n" + "="*70)
        print("🚀 Preparing Yelp Dataset for Federated Learning")
        print("="*70)
        
        # Step 1: Load data
        if not self.load_yelp_data():
            return None, None
        
        # Step 2: Prepare interactions
        df_interactions = self.prepare_interaction_data()
        
        # Step 3: Preprocess
        user_ids, item_ids, ratings, text_features, metadata, df_filtered = \
            self.preprocess_for_federated_learning(df_interactions)
        
        # Step 4: Split for clients
        client_data = self.split_for_federated_learning(
            user_ids, item_ids, ratings, text_features, num_clients
        )
        
        # Step 5: Save
        self.save_processed_data(client_data, metadata)
        
        print("\n" + "="*70)
        print("✅ Dataset preparation complete!")
        print("="*70)
        
        return client_data, metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Yelp dataset for federated learning")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    
    args = parser.parse_args()
    
    # Prepare dataset
    prep = YelpDatasetPreparation(data_dir=args.data_dir)
    client_data, metadata = prep.prepare_yelp_for_federated_learning(num_clients=args.num_clients)
    
    if client_data and metadata:
        print("\n📊 Summary:")
        print(f"   Total users: {metadata['num_users']}")
        print(f"   Total items: {metadata['num_items']}")
        print(f"   Total images: {metadata['num_images']}")
        print(f"   Clients: {len(client_data)}")
