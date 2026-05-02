#!/usr/bin/env python3
"""
Download Amazon Reviews 2023 dataset with REAL product images
Source: McAuley-Lab/Amazon-Reviews-2023 on Hugging Face
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import argparse
import json

def download_image(url: str, save_path: str, timeout: int = 15):
    """Download a single image"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        pass
    return False

def download_amazon_real(data_dir: str, category: str = "All_Beauty", 
                         max_reviews: int = 10000, max_images: int = 1000):
    """
    Download Amazon Reviews 2023 with real product images
    
    Args:
        data_dir: Base data directory
        category: Product category (All_Beauty, Electronics, Books, etc.)
        max_reviews: Maximum number of reviews to download
        max_images: Maximum number of images to download
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets library: pip install datasets")
        return False
    
    # Create directories
    raw_dir = Path(data_dir) / "raw" / f"amazon_{category.lower()}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir = raw_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    print(f"\n📦 Downloading Amazon Reviews 2023 - {category}")
    print(f"   Reviews: {max_reviews}")
    print(f"   Images: {max_images}")
    print("\n⏳ This may take a few minutes...\n")
    
    # Load reviews dataset
    print("1️⃣ Loading reviews...")
    try:
        reviews_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_review_{category}", 
            trust_remote_code=True,
            streaming=True
        )
        
        # Collect reviews
        reviews = []
        for i, review in enumerate(tqdm(reviews_dataset["full"], total=max_reviews, desc="Reviews")):
            if i >= max_reviews:
                break
            reviews.append({
                'user_id': review.get('user_id', ''),
                'item_id': review.get('parent_asin', review.get('asin', '')),
                'rating': review.get('rating', 0),
                'review_title': review.get('title', ''),
                'review_text': review.get('text', ''),
                'timestamp': review.get('timestamp', 0),
                'verified_purchase': review.get('verified_purchase', False),
                'helpful_vote': review.get('helpful_vote', 0)
            })
        
        df_reviews = pd.DataFrame(reviews)
        df_reviews.to_csv(raw_dir / "reviews.csv", index=False)
        print(f"✅ Saved {len(df_reviews)} reviews\n")
        
    except Exception as e:
        print(f"❌ Error loading reviews: {e}")
        return False
    
    # Load product metadata
    print("2️⃣ Loading product metadata...")
    try:
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            trust_remote_code=True,
            streaming=True
        )
        
        # Get unique items from reviews
        unique_items = set(df_reviews['item_id'].unique())
        
        # Collect metadata
        products = []
        image_urls = {}
        
        for product in tqdm(meta_dataset["full"], desc="Products"):
            parent_asin = product.get('parent_asin')
            if parent_asin in unique_items:
                products.append({
                    'item_id': parent_asin,
                    'title': product.get('title', ''),
                    'main_category': product.get('main_category', category),
                    'average_rating': product.get('average_rating', 0),
                    'rating_number': product.get('rating_number', 0),
                    'store': product.get('store', ''),
                    'description': str(product.get('description', ''))[:500]  # Limit length
                })
                
                # Extract image URLs
                images = product.get('images', {})
                if images and 'large' in images and images['large']:
                    image_urls[parent_asin] = images['large']
            
            if len(products) >= len(unique_items):
                break
        
        df_products = pd.DataFrame(products)
        df_products.to_csv(raw_dir / "products.csv", index=False)
        print(f"✅ Saved {len(df_products)} products\n")
        
    except Exception as e:
        print(f"❌ Error loading products: {e}")
        df_products = pd.DataFrame()
        image_urls = {}
    
    # Download images
    print(f"3️⃣ Downloading {max_images} real product images...")
    downloaded = 0
    failed = 0
    
    # Limit to max_images
    items_to_download = list(image_urls.items())[:max_images]
    
    for item_id, urls in tqdm(items_to_download, desc="Images"):
        if urls and len(urls) > 0:
            # Download first available image
            for url in urls:
                ext = url.split('.')[-1].split('?')[0]
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                
                save_path = image_dir / f"{item_id}.{ext}"
                
                if not save_path.exists():
                    if download_image(url, str(save_path)):
                        downloaded += 1
                        break
                else:
                    downloaded += 1
                    break
            else:
                failed += 1
    
    print(f"✅ Downloaded {downloaded} real product images")
    if failed > 0:
        print(f"⚠️  Failed to download {failed} images\n")
    
    # Print summary
    print("="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    print(f"Category: {category}")
    print(f"Reviews: {len(df_reviews):,}")
    print(f"Products: {len(df_products):,}")
    print(f"Images: {downloaded:,} real product photos")
    print(f"Location: {raw_dir}")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in raw_dir.rglob('*') if f.is_file())
    print(f"Total Size: {total_size / (1024*1024):.1f} MB")
    print("="*70)
    print("✅ REAL multimodal Amazon dataset ready!")
    print("   - Real user reviews")
    print("   - Real product images from Amazon")
    print("   - Perfect for multimodal recommendation!")
    print("="*70)
    
    return True

def list_categories():
    """List available Amazon product categories"""
    categories = [
        ("All_Beauty", "Beauty products (~smaller, good for testing)"),
        ("Electronics", "Electronics (~larger dataset)"),
        ("Books", "Books (~large dataset)"),
        ("Home_and_Kitchen", "Home & Kitchen (~medium dataset)"),
        ("Sports_and_Outdoors", "Sports & Outdoors (~medium dataset)"),
        ("Toys_and_Games", "Toys & Games (~medium dataset)"),
        ("Clothing_Shoes_and_Jewelry", "Fashion (~large dataset)"),
        ("Movies_and_TV", "Movies & TV (~large dataset)"),
        ("Pet_Supplies", "Pet Supplies (~smaller)"),
        ("Automotive", "Automotive (~smaller)"),
    ]
    
    print("\n📋 AVAILABLE CATEGORIES:")
    print("-" * 50)
    for cat, desc in categories:
        print(f"   {cat:<30} - {desc}")
    print("-" * 50)
    print("\n💡 Tip: Start with 'All_Beauty' for faster testing!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 with real images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small test dataset (fast)
  python download_amazon_real.py --category All_Beauty --max-reviews 1000 --max-images 100
  
  # Medium dataset (good for testing)
  python download_amazon_real.py --category All_Beauty --max-reviews 5000 --max-images 500
  
  # Large dataset (full training)
  python download_amazon_real.py --category Electronics --max-reviews 50000 --max-images 5000
        """
    )
    
    parser.add_argument("--data-dir", default="data", 
                       help="Data directory (default: data)")
    parser.add_argument("--category", default="All_Beauty",
                       help="Product category (default: All_Beauty)")
    parser.add_argument("--max-reviews", type=int, default=10000,
                       help="Maximum reviews to download (default: 10000)")
    parser.add_argument("--max-images", type=int, default=1000,
                       help="Maximum images to download (default: 1000)")
    parser.add_argument("--list-categories", action="store_true",
                       help="List all available categories")
    
    args = parser.parse_args()
    
    if args.list_categories:
        list_categories()
    else:
        success = download_amazon_real(
            data_dir=args.data_dir,
            category=args.category,
            max_reviews=args.max_reviews,
            max_images=args.max_images
        )
        
        if not success:
            print("\n❌ Download failed. Try a different category or smaller size.")
            print("   Run with --list-categories to see available options.")
