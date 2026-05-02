#!/usr/bin/env python3
"""
Download REAL Yelp Multimodal Recommendation dataset with actual photos
This dataset has real Yelp photos, reviews, and business data
Size: ~743 MB (much smaller than official 12GB dataset)
"""

import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import argparse
import urllib.request

def download_file_with_progress(url: str, destination: str):
    """Download file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def download_yelp_multimodal_real(data_dir: str, max_reviews: int = 10000, max_photos: int = 1000):
    """
    Download REAL Yelp Multimodal dataset with actual photos from Hugging Face
    
    Args:
        data_dir: Base data directory
        max_reviews: Maximum number of reviews to keep
        max_photos: Maximum number of photos to download
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets library: pip install datasets")
        return False
    
    # Create directories
    raw_dir = Path(data_dir) / "raw" / "yelp_multimodal_real"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir = raw_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    print(f"\n📦 Downloading REAL Yelp Multimodal Dataset")
    print(f"   Source: Hugging Face (wzehui/Yelp-Multimodal-Recommendation)")
    print(f"   Total Size: ~743 MB")
    print(f"   You'll get: {max_reviews} reviews + {max_photos} real photos\n")
    
    # Load the dataset
    print("⏳ Loading dataset from Hugging Face...")
    print("   (This may take a few minutes for the full 743MB dataset)")
    
    try:
        # Try to load the dataset
        dataset = load_dataset("wzehui/Yelp-Multimodal-Recommendation", streaming=True)
        print("✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Alternative: Download from direct links")
        print("   I'll provide alternative download methods...")
        return False
    
    # Process and save subsets
    print(f"\n⏳ Processing {max_reviews} reviews...")
    
    # Extract reviews
    reviews = []
    for i, item in enumerate(dataset["train"]):
        if i >= max_reviews:
            break
        reviews.append(item)
    
    df_reviews = pd.DataFrame(reviews)
    df_reviews.to_csv(raw_dir / "reviews_subset.csv", index=False)
    print(f"✅ Saved {len(df_reviews)} reviews")
    
    # Download photos
    if 'photo_url' in df_reviews.columns or 'image_url' in df_reviews.columns:
        print(f"\n⏳ Downloading {max_photos} real photos...")
        
        photo_col = 'photo_url' if 'photo_url' in df_reviews.columns else 'image_url'
        photos_to_download = df_reviews[photo_col].dropna().unique()[:max_photos]
        
        downloaded = 0
        for i, url in enumerate(tqdm(photos_to_download, desc="Downloading photos")):
            try:
                ext = url.split('.')[-1].split('?')[0]
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                
                save_path = image_dir / f"photo_{i}.{ext}"
                
                if not save_path.exists():
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        downloaded += 1
            except:
                continue
        
        print(f"✅ Downloaded {downloaded} real photos")
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Reviews: {len(df_reviews)}")
    print(f"   Location: {raw_dir}")
    
    total_size = sum(f.stat().st_size for f in raw_dir.rglob('*') if f.is_file())
    print(f"   Total Size: {total_size / (1024*1024):.1f} MB")
    
    print(f"\n✅ REAL multimodal dataset ready!")
    print(f"   You now have actual Yelp photos + real reviews!")
    
    return True

def manual_download_instructions():
    """Provide manual download instructions if automatic download fails"""
    print("\n" + "="*70)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nOption 1: Hugging Face CLI (Recommended)")
    print("   1. Install Hugging Face CLI:")
    print("      pip install huggingface-hub")
    print("\n   2. Download the dataset:")
    print("      huggingface-cli download wzehui/Yelp-Multimodal-Recommendation --local-dir data/raw/yelp_multimodal_real")
    print("\n   3. Create a subset manually using the provided scripts")
    
    print("\nOption 2: Direct Browser Download")
    print("   1. Visit: https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation")
    print("   2. Click 'Files and versions'")
    print("   3. Download the files you need:")
    print("      - business.csv")
    print("      - review.csv")
    print("      - photo.csv")
    print("   4. Extract and place in: data/raw/yelp_multimodal_real/")
    
    print("\nOption 3: Git LFS Download")
    print("   1. Install git-lfs:")
    print("      git lfs install")
    print("\n   2. Clone the dataset:")
    print("      git clone https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation")
    print("      mv Yelp-Multimodal-Recommendation data/raw/yelp_multimodal_real")
    
    print("\n" + "="*70)
    print("💡 After downloading, run the subset creation script:")
    print("   python data/create_yelp_subset.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download REAL Yelp Multimodal dataset")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max-reviews", type=int, default=10000, help="Maximum reviews")
    parser.add_argument("--max-photos", type=int, default=1000, help="Maximum photos to download")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions")
    
    args = parser.parse_args()
    
    if args.manual:
        manual_download_instructions()
    else:
        success = download_yelp_multimodal_real(
            data_dir=args.data_dir,
            max_reviews=args.max_reviews,
            max_photos=args.max_photos
        )
        
        if not success:
            manual_download_instructions()
