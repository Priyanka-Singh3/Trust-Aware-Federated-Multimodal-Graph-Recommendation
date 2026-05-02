#!/usr/bin/env python3
"""
Create a multimodal sample from Hugging Face Yelp dataset with dummy images
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
import random
from pathlib import Path

def create_multimodal_sample(data_dir: str, num_samples: int = 10000, image_size: tuple = (224, 224)):
    """Create multimodal sample from Hugging Face Yelp dataset"""
    
    # Load Hugging Face dataset
    hf_dir = Path(data_dir) / "raw" / "yelp_hf"
    sample_dir = Path(data_dir) / "raw" / "yelp_multimodal_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {hf_dir}")
    
    # Load train data
    df_train = pd.read_csv(hf_dir / "train.csv")
    df_test = pd.read_csv(hf_dir / "test.csv")
    
    # Combine and sample
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_sample = df_all.sample(n=min(num_samples, len(df_all)), random_state=42)
    
    # Create user and item IDs
    df_sample['user_id'] = [f"user_{i}" for i in range(len(df_sample))]
    df_sample['item_id'] = [f"business_{i}" for i in range(len(df_sample))]
    df_sample['rating'] = df_sample['label']  # Convert label (1-5) to rating
    df_sample['review_text'] = df_sample['text']
    df_sample['timestamp'] = [random.randint(1609459200, 1672531199) for _ in range(len(df_sample))]
    
    # Select columns for compatibility with existing code
    df_final = df_sample[['user_id', 'item_id', 'rating', 'review_text', 'timestamp']]
    
    # Save reviews data
    df_final.to_csv(sample_dir / "reviews_sample.csv", index=False)
    print(f"Created sample with {len(df_final)} reviews")
    
    # Create dummy images for businesses
    image_dir = sample_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    unique_items = df_final['item_id'].unique()
    print(f"Creating {len(unique_items)} dummy images...")
    
    for item_id in unique_items:
        # Create random colored image
        img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        image_path = image_dir / f"{item_id}.jpg"
        img.save(image_path)
    
    print(f"Created {len(unique_items)} dummy images in {image_dir}")
    
    # Create business metadata
    business_data = []
    categories = ['Restaurant', 'Shopping', 'Health', 'Entertainment', 'Automotive', 'Beauty', 'Home']
    
    for item_id in unique_items:
        business_data.append({
            'business_id': item_id,
            'name': f"Business {item_id.split('_')[1]}",
            'categories': [random.choice(categories)],
            'stars': round(random.uniform(3.0, 5.0), 1),
            'review_count': random.randint(10, 500)
        })
    
    df_businesses = pd.DataFrame(business_data)
    df_businesses.to_csv(sample_dir / "businesses_sample.csv", index=False)
    print(f"Created business metadata for {len(df_businesses)} businesses")
    
    # Create user metadata
    unique_users = df_final['user_id'].unique()
    user_data = []
    
    for user_id in unique_users:
        user_data.append({
            'user_id': user_id,
            'review_count': random.randint(1, 50),
            'average_stars': round(random.uniform(3.0, 5.0), 1),
            'friends': random.sample(list(unique_users), min(random.randint(0, 10), len(unique_users)))
        })
    
    df_users = pd.DataFrame(user_data)
    df_users.to_csv(sample_dir / "users_sample.csv", index=False)
    print(f"Created user metadata for {len(df_users)} users")
    
    print(f"\nMultimodal sample created successfully!")
    print(f"Location: {sample_dir}")
    print(f"Files created:")
    print(f"  - reviews_sample.csv ({len(df_final)} reviews)")
    print(f"  - businesses_sample.csv ({len(df_businesses)} businesses)")
    print(f"  - users_sample.csv ({len(df_users)} users)")
    print(f"  - images/ ({len(unique_items)} business images)")
    print(f"\nTotal size: ~{sum(f.stat().st_size for f in sample_dir.rglob('*') if f.is_file()) / (1024*1024):.1f} MB")
    
    return sample_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create multimodal sample from Yelp dataset")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of reviews to sample")
    
    args = parser.parse_args()
    
    create_multimodal_sample(args.data_dir, args.num_samples)
