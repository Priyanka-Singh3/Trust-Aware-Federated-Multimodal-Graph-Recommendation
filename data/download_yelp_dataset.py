#!/usr/bin/env python3
"""
Script to download and setup Yelp dataset for the federated multimodal recommendation system
"""

import os
import requests
import zipfile
import tarfile
import json
import pandas as pd
from pathlib import Path
import argparse

def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download file with progress bar"""
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
    
    print(f"\nDownloaded {downloaded / (1024*1024):.1f} MB")

def download_yelp_official(data_dir: str, include_photos: bool = False):
    """Download Yelp dataset from official source"""
    
    raw_dir = Path(data_dir) / "raw" / "yelp"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download JSON data
    json_url = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
    json_path = raw_dir / "Yelp-JSON.zip"
    
    if not json_path.exists():
        download_file(json_url, str(json_path))
        
        # Extract JSON files
        print("Extracting JSON files...")
        with zipfile.ZipFile(json_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
    else:
        print("JSON data already exists")
    
    # Download photos if requested
    if include_photos:
        photos_url = "https://business.yelp.com/external-assets/files/Yelp-Photos.zip"
        photos_path = raw_dir / "Yelp-Photos.zip"
        
        if not photos_path.exists():
            download_file(photos_url, str(photos_path))
            
            # Extract photos
            print("Extracting photos...")
            with zipfile.ZipFile(photos_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
        else:
            print("Photos already exists")

def download_yelp_huggingface(data_dir: str):
    """Download Yelp dataset from Hugging Face"""
    try:
        from datasets import load_dataset
        
        print("Loading Yelp dataset from Hugging Face...")
        dataset = load_dataset("Yelp/yelp_review_full")
        
        # Save to local directory
        raw_dir = Path(data_dir) / "raw" / "yelp_hf"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train and test splits
        dataset["train"].to_csv(raw_dir / "train.csv", index=False)
        dataset["test"].to_csv(raw_dir / "test.csv", index=False)
        
        print(f"Dataset saved to {raw_dir}")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        
        return True
        
    except ImportError:
        print("Hugging Face datasets library not installed. Run: pip install datasets")
        return False
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        return False

def setup_yelp_kaggle(data_dir: str, kaggle_path: str):
    """Setup Yelp dataset from Kaggle (requires manual download)"""
    
    raw_dir = Path(data_dir) / "raw" / "yelp_kaggle"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract if it's a zip/tar file
    if kaggle_path.endswith('.zip'):
        with zipfile.ZipFile(kaggle_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
    elif kaggle_path.endswith('.tar') or kaggle_path.endswith('.tar.gz'):
        with tarfile.open(kaggle_path, 'r:*') as tar_ref:
            tar_ref.extractall(raw_dir)
    else:
        # Assume it's already extracted
        import shutil
        shutil.copy2(kaggle_path, raw_dir)
    
    print(f"Kaggle dataset setup in {raw_dir}")

def create_sample_from_yelp(data_dir: str, num_samples: int = 10000, include_businesses: bool = True, include_users: bool = True):
    """Create a smaller sample from the full Yelp dataset for testing"""
    
    raw_dir = Path(data_dir) / "raw" / "yelp"
    sample_dir = Path(data_dir) / "raw" / "yelp_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample dataset with {num_samples} reviews...")
    
    # Load and sample reviews
    review_file = raw_dir / "yelp_academic_dataset_review.json"
    if not review_file.exists():
        print(f"Review file not found: {review_file}")
        return False
    
    reviews = []
    business_ids = set()
    user_ids = set()
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples * 2:  # Read more to ensure we get enough after filtering
                break
            try:
                review = json.loads(line)
                reviews.append(review)
                business_ids.add(review['business_id'])
                user_ids.add(review['user_id'])
            except json.JSONDecodeError:
                continue
    
    # Take only requested number of reviews
    reviews = reviews[:num_samples]
    
    # Convert to DataFrame and save reviews
    df_reviews = pd.DataFrame(reviews)
    df_reviews.to_csv(sample_dir / "reviews_sample.csv", index=False)
    
    # Save reviews as JSON
    with open(sample_dir / "reviews_sample.json", 'w', encoding='utf-8') as f:
        for review in reviews:
            f.write(json.dumps(review) + '\n')
    
    print(f"Sample dataset created with {len(reviews)} reviews")
    
    # Extract related businesses if requested
    if include_businesses and business_ids:
        business_file = raw_dir / "yelp_academic_dataset_business.json"
        if business_file.exists():
            businesses = []
            with open(business_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        business = json.loads(line)
                        if business['business_id'] in business_ids:
                            businesses.append(business)
                    except json.JSONDecodeError:
                        continue
            
            df_businesses = pd.DataFrame(businesses)
            df_businesses.to_csv(sample_dir / "businesses_sample.csv", index=False)
            print(f"Included {len(businesses)} related businesses")
    
    # Extract related users if requested
    if include_users and user_ids:
        user_file = raw_dir / "yelp_academic_dataset_user.json"
        if user_file.exists():
            users = []
            with open(user_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        user = json.loads(line)
                        if user['user_id'] in user_ids:
                            users.append(user)
                    except json.JSONDecodeError:
                        continue
            
            df_users = pd.DataFrame(users)
            df_users.to_csv(sample_dir / "users_sample.csv", index=False)
            print(f"Included {len(users)} related users")
    
    print(f"Sample dataset saved to {sample_dir}")
    print("Files created:")
    print(f"  - reviews_sample.csv ({len(reviews)} reviews)")
    if include_businesses:
        print(f"  - businesses_sample.csv")
    if include_users:
        print(f"  - users_sample.csv")
    
    return True

def download_subset_photos(data_dir: str, num_photos: int = 1000):
    """Download only a subset of photos to save space"""
    
    raw_dir = Path(data_dir) / "raw" / "yelp"
    sample_dir = Path(data_dir) / "raw" / "yelp_sample"
    photos_dir = sample_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    
    # First download the full photos zip
    photos_url = "https://business.yelp.com/external-assets/files/Yelp-Photos.zip"
    photos_zip = raw_dir / "Yelp-Photos.zip"
    
    if not photos_zip.exists():
        print("Downloading photos zip file (this may take a while)...")
        download_file(photos_url, str(photos_zip))
    
    # Extract only a subset of photos
    print(f"Extracting {num_photos} sample photos...")
    
    with zipfile.ZipFile(photos_zip, 'r') as zip_ref:
        photo_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]
        selected_photos = photo_files[:num_photos]
        
        for photo_file in selected_photos:
            try:
                zip_ref.extract(photo_file, photos_dir)
            except:
                continue
    
    print(f"Extracted {len(selected_photos)} photos to {photos_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and setup Yelp dataset")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--source", choices=["official", "huggingface", "kaggle"], 
                       default="official", help="Dataset source")
    parser.add_argument("--include-photos", action="store_true", help="Download photos (official only)")
    parser.add_argument("--kaggle-path", help="Path to Kaggle dataset file")
    parser.add_argument("--create-sample", type=int, help="Create sample with N reviews")
    parser.add_argument("--download-photos-subset", type=int, help="Download subset of N photos")
    
    args = parser.parse_args()
    
    if args.source == "official":
        download_yelp_official(args.data_dir, args.include_photos)
    elif args.source == "huggingface":
        download_yelp_huggingface(args.data_dir)
    elif args.source == "kaggle":
        if not args.kaggle_path:
            print("Please provide --kaggle-path for Kaggle source")
            return
        setup_yelp_kaggle(args.data_dir, args.kaggle_path)
    
    if args.create_sample:
        create_sample_from_yelp(args.data_dir, args.create_sample)
    
    if args.download_photos_subset:
        download_subset_photos(args.data_dir, args.download_photos_subset)

if __name__ == "__main__":
    main()
