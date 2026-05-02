#!/usr/bin/env python3
"""
Extract only needed files from wzehui/Yelp-Multimodal-Recommendation
and create a clean subset
"""

import pandas as pd
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

def extract_clean_dataset(data_dir: str, max_records: int = 10000):
    """
    Download and extract clean Yelp Multimodal dataset
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets library: pip install datasets")
        return False
    
    # Create output directory
    output_dir = Path(data_dir) / "raw" / "yelp_multimodal_clean"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📦 Extracting Yelp Multimodal Dataset")
    print("="*70)
    print(f"   Dataset: wzehui/Yelp-Multimodal-Recommendation")
    print(f"   Max Records: {max_records:,}")
    print()
    
    # Download full dataset
    print("⏳ Step 1: Loading dataset from Hugging Face...")
    print("   (Downloading ~743MB to cache)")
    
    try:
        dataset = load_dataset("wzehui/Yelp-Multimodal-Recommendation", download_mode="force_redownload")
        print("✅ Dataset loaded!\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Check available splits
    print("📁 Available splits:", list(dataset.keys()))
    
    # Use train split
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    full_dataset = dataset[split_name]
    
    print(f"📊 Full dataset size: {len(full_dataset):,} records\n")
    
    # Extract subset
    print(f"2️⃣ Extracting {max_records:,} records...")
    records = []
    
    for i in tqdm(range(min(max_records, len(full_dataset)))):
        records.append(full_dataset[i])
    
    df = pd.DataFrame(records)
    
    # Save main dataset
    main_file = output_dir / "yelp_data.csv"
    df.to_csv(main_file, index=False)
    print(f"✅ Saved main data: {main_file}")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {', '.join(df.columns.tolist())}\n")
    
    # Check for photo data
    photo_cols = [col for col in df.columns if 'photo' in col.lower() or 'image' in col.lower()]
    if photo_cols:
        print(f"📸 Photo columns found: {photo_cols}")
        
        # Save photo references
        for col in photo_cols:
            if col in df.columns:
                photo_data = df[['business_id' if 'business_id' in df.columns else df.columns[0], col]].dropna()
                if len(photo_data) > 0:
                    photo_file = output_dir / f"{col}_data.csv"
                    photo_data.to_csv(photo_file, index=False)
                    print(f"✅ Saved {col}: {len(photo_data)} entries")
    
    # Create sample with essential columns only
    essential_cols = ['business_id', 'name', 'categories', 'stars', 'review_count', 'city', 'state']
    available_cols = [col for col in essential_cols if col in df.columns]
    
    if available_cols:
        df_essential = df[available_cols].drop_duplicates()
        essential_file = output_dir / "businesses_clean.csv"
        df_essential.to_csv(essential_file, index=False)
        print(f"✅ Saved clean businesses: {len(df_essential):,} unique\n")
    
    # Print summary
    print("="*70)
    print("📊 CLEAN DATASET SUMMARY")
    print("="*70)
    
    total_size = 0
    for f in output_dir.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024*1024)
            total_size += f.stat().st_size
            print(f"{f.name:<30} {size_mb:>8.1f} MB")
    
    print("-"*70)
    print(f"{'TOTAL':<30} {total_size / (1024*1024):>8.1f} MB")
    print("="*70)
    print(f"\n📂 Location: {output_dir}")
    
    # Cache info
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    print(f"\n💾 Full cache: {cache_dir}")
    print("   To delete cache and save space:")
    print(f"   rm -rf {cache_dir}/wzehui___yelp-multimodal-recommendation")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clean Yelp Multimodal dataset"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max-records", type=int, default=10000, 
                       help="Maximum records to extract")
    
    args = parser.parse_args()
    
    extract_clean_dataset(
        data_dir=args.data_dir,
        max_records=args.max_records
    )
