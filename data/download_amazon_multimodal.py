#!/usr/bin/env python3
"""
Download Amazon Reviews 2023 dataset with real product images
Best multimodal dataset for recommendation systems
"""

import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

def download_image(url: str, save_path: str):
    """Download a single image"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def download_amazon_multimodal(data_dir: str, category: str = "All_Beauty", max_reviews: int = 10000, max_images: int = 1000):
    """
    Download Amazon Reviews 2023 dataset with real images
    
    Args:
        data_dir: Base data directory
        category: Product category (All_Beauty, Electronics, Books, etc.)
        max_reviews: Maximum number of reviews to download
        max_images: Maximum number of images to download
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets library: pip install datasets")
        return False
    
    # Create directories
    raw_dir = Path(data_dir) / "raw" / f"amazon_{category.lower()}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir = raw_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    print(f"\n📦 Downloading Amazon Reviews 2023 - {category}")
    print(f"   Reviews: {max_reviews}")
    print(f"   Images: {max_images}")
    
    # Load reviews dataset
    print("\n⏳ Loading reviews...")
    reviews_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", 
        f"raw_review_{category}", 
        trust_remote_code=True,
        streaming=True
    )
    
    # Collect reviews
    reviews = []
    for i, review in enumerate(reviews_dataset["full"]):
        if i >= max_reviews:
            break
        reviews.append({
            'user_id': review.get('user_id', ''),
            'item_id': review.get('parent_asin', review.get('asin', '')),
            'rating': review.get('rating', 0),
            'review_text': review.get('text', ''),
            'review_title': review.get('title', ''),
            'timestamp': review.get('timestamp', 0),
            'verified_purchase': review.get('verified_purchase', False),
            'helpful_vote': review.get('helpful_vote', 0)
        })
    
    df_reviews = pd.DataFrame(reviews)
    df_reviews.to_csv(raw_dir / "reviews.csv", index=False)
    print(f"✅ Downloaded {len(df_reviews)} reviews")
    
    # Load product metadata with images
    print("\n⏳ Loading product metadata...")
    meta_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        trust_remote_code=True,
        streaming=True
    )
    
    # Get unique items from reviews
    unique_items = set(df_reviews['item_id'].unique())
    
    # Collect metadata for items in our review set
    products = []
    image_urls = {}  # item_id -> image URLs
    
    for product in meta_dataset["full"]:
        parent_asin = product.get('parent_asin')
        if parent_asin in unique_items:
            products.append({
                'item_id': parent_asin,
                'title': product.get('title', ''),
                'main_category': product.get('main_category', ''),
                'average_rating': product.get('average_rating', 0),
                'rating_number': product.get('rating_number', 0),
                'store': product.get('store', ''),
                'description': str(product.get('description', ''))
            })
            
            # Extract image URLs
            images = product.get('images', {})
            if images and 'large' in images and images['large']:
                image_urls[parent_asin] = images['large']
        
        if len(products) >= len(unique_items):
            break
    
    df_products = pd.DataFrame(products)
    df_products.to_csv(raw_dir / "products.csv", index=False)
    print(f"✅ Downloaded {len(df_products)} product metadata")
    
    # Download images
    print(f"\n⏳ Downloading up to {max_images} product images...")
    downloaded_count = 0
    
    for item_id, urls in tqdm(list(image_urls.items())[:max_images]):
        if urls and len(urls) > 0:
            # Download first image for each product
            image_url = urls[0]
            ext = image_url.split('.')[-1].split('?')[0]  # Get extension
            if ext not in ['jpg', 'jpeg', 'png']:
                ext = 'jpg'
            
            save_path = image_dir / f"{item_id}.{ext}"
            
            if not save_path.exists():  # Skip if already exists
                if download_image(image_url, str(save_path)):
                    downloaded_count += 1
    
    print(f"✅ Downloaded {downloaded_count} real product images")
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Reviews: {len(df_reviews)}")
    print(f"   Products: {len(df_products)}")
    print(f"   Images: {downloaded_count}")
    print(f"   Location: {raw_dir}")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in raw_dir.rglob('*') if f.is_file())
    print(f"   Total Size: {total_size / (1024*1024):.1f} MB")
    
    return True

def list_available_categories():
    """List available Amazon product categories"""
    categories = [
        "All_Beauty",
        "Amazon_Fashion",
        "Appliances",
        "Arts_Crafts_and_Sewing",
        "Automotive",
        "Books",
        "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",
        "Electronics",
        "Gift_Cards",
        "Grocery_and_Gourmet_Food",
        "Home_and_Kitchen",
        "Industrial_and_Scientific",
        "Kindle_Store",
        "Luxury_Beauty",
        "Magazine_Subscriptions",
        "Movies_and_TV",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Prime_Pantry",
        "Software",
        "Sports_and_Outdoors",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games"
    ]
    return categories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 with real images")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--category", default="All_Beauty", 
                       help="Product category (use --list-categories to see all)")
    parser.add_argument("--max-reviews", type=int, default=10000, help="Maximum reviews to download")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum images to download")
    parser.add_argument("--list-categories", action="store_true", help="List available categories")
    
    args = parser.parse_args()
    
    if args.list_categories:
        print("\n📋 Available Categories:")
        for cat in list_available_categories():
            print(f"   - {cat}")
        print("\n💡 Tip: Start with smaller categories like 'All_Beauty' or 'Books' for testing")
    else:
        download_amazon_multimodal(
            data_dir=args.data_dir,
            category=args.category,
            max_reviews=args.max_reviews,
            max_images=args.max_images
        )
