#!/usr/bin/env python3
"""
Download Yelp Multimodal dataset using datasets library
Extract only needed files and remove extra data to save space
"""

import pandas as pd
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

def download_yelp_clean(data_dir: str, max_reviews: int = 10000, max_photos: int = 1000):
    """
    Download Yelp Multimodal dataset and extract only needed files
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets library: pip install datasets")
        return False
    
    # Create output directory
    output_dir = Path(data_dir) / "raw" / "yelp_clean"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📦 Downloading Yelp Multimodal Dataset (Clean Version)")
    print("="*70)
    print(f"   Max Reviews: {max_reviews}")
    print(f"   Max Photos: {max_photos}")
    print("\n⏳ Loading dataset from Hugging Face...")
    print("   (This will download ~743MB to cache)")
    
    try:
        # Load the dataset
        dataset = load_dataset("wzehui/Yelp-Multimodal-Recommendation")
        print("✅ Dataset loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Check what's available
    print("📁 Available splits:", list(dataset.keys()))
    print("Sample item:", dataset['train'][0] if 'train' in dataset else "N/A")
    print()
    
    # Extract reviews
    print(f"1️⃣ Extracting {max_reviews} reviews...")
    reviews = []
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    
    for i, item in enumerate(tqdm(dataset[split_name], total=min(max_reviews, len(dataset[split_name])))):
        if i >= max_reviews:
            break
        reviews.append(item)
    
    df_reviews = pd.DataFrame(reviews)
    reviews_file = output_dir / "reviews.csv"
    df_reviews.to_csv(reviews_file, index=False)
    print(f"✅ Saved {len(df_reviews)} reviews to {reviews_file}\n")
    
    # Check for photo URLs
    photo_urls = []
    if 'photo_url' in df_reviews.columns:
        photo_urls = df_reviews['photo_url'].dropna().unique().tolist()
    elif 'image_url' in df_reviews.columns:
        photo_urls = df_reviews['image_url'].dropna().unique().tolist()
    
    if photo_urls:
        print(f"2️⃣ Found {len(photo_urls)} photo URLs")
        # Save photo URLs for later download
        with open(output_dir / "photo_urls.txt", "w") as f:
            for url in photo_urls[:max_photos]:
                f.write(f"{url}\n")
        print(f"✅ Saved {min(len(photo_urls), max_photos)} photo URLs\n")
    
    # Print summary
    print("="*70)
    print("📊 CLEAN DATASET SUMMARY")
    print("="*70)
    print(f"Reviews: {len(df_reviews):,}")
    print(f"Columns: {', '.join(df_reviews.columns.tolist())}")
    print(f"Location: {output_dir}")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"Total Size: {total_size / (1024*1024):.1f} MB")
    print("="*70)
    print("✅ Clean dataset ready!")
    print("="*70)
    
    # Show cache location and cleanup info
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    print(f"\n💾 Full dataset cached at: {cache_dir}")
    print("   (You can delete this later to save space)")
    print(f"\n📂 Your clean dataset is at: {output_dir}")
    
    return True

def cleanup_cache():
    """Remove huggingface cache to save space"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    
    if cache_dir.exists():
        print(f"\n🧹 Cleaning up cache at {cache_dir}...")
        try:
            shutil.rmtree(cache_dir)
            print("✅ Cache cleaned!")
        except Exception as e:
            print(f"⚠️  Could not clean cache: {e}")
    else:
        print("\n✅ No cache to clean")

def show_cache_info():
    """Show cache location and size"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    
    if cache_dir.exists():
        size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        print(f"\n💾 Cache location: {cache_dir}")
        print(f"   Cache size: {size / (1024*1024*1024):.1f} GB")
        print("\n   To clean cache manually:")
        print(f"   rm -rf {cache_dir}")
    else:
        print("\n✅ No cache found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download clean Yelp Multimodal dataset"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max-reviews", type=int, default=10000, help="Maximum reviews")
    parser.add_argument("--max-photos", type=int, default=1000, help="Maximum photos")
    parser.add_argument("--cleanup", action="store_true", help="Clean huggingface cache after download")
    parser.add_argument("--cache-info", action="store_true", help="Show cache info only")
    
    args = parser.parse_args()
    
    if args.cache_info:
        show_cache_info()
    else:
        success = download_yelp_clean(
            data_dir=args.data_dir,
            max_reviews=args.max_reviews,
            max_photos=args.max_photos
        )
        
        if success and args.cleanup:
            cleanup_cache()
