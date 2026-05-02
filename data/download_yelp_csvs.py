#!/usr/bin/env python3
"""
Download wzehui/Yelp-Multimodal-Recommendation CSV files directly
Extract only needed columns and save clean dataset
"""

import pandas as pd
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

def download_yelp_csvs(data_dir: str, max_records: int = 10000):
    """
    Download and process Yelp Multimodal CSV files
    """
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets library: pip install datasets")
        return False
    
    # Create output directory
    output_dir = Path(data_dir) / "raw" / "yelp_multimodal_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📦 Downloading Yelp Multimodal CSV Files")
    print("="*70)
    
    # Define the CSV files we want
    csv_files = {
        'business': 'business.csv',
        'review': 'review.csv', 
        'photo': 'photo.csv',
        'checkin': 'checkin.csv'
    }
    
    downloaded_files = {}
    
    # Download each CSV file separately
    for name, filename in csv_files.items():
        print(f"\n⏳ Downloading {filename}...")
        try:
            # Load specific file
            dataset = load_dataset(
                "wzehui/Yelp-Multimodal-Recommendation", 
                data_files=filename,
                download_mode="reuse_cache_if_exists"
            )
            
            # Get the data
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            
            print(f"   Total records: {len(data):,}")
            
            # Extract subset
            records = []
            limit = min(max_records, len(data))
            for i in tqdm(range(limit), desc=f"Extracting {name}"):
                records.append(data[i])
            
            # Save to CSV
            df = pd.DataFrame(records)
            output_file = output_dir / f"{name}_clean.csv"
            df.to_csv(output_file, index=False)
            
            downloaded_files[name] = {
                'file': output_file,
                'records': len(df),
                'columns': df.columns.tolist()
            }
            
            print(f"✅ Saved: {output_file}")
            print(f"   Records: {len(df):,}")
            
        except Exception as e:
            print(f"⚠️  Could not download {filename}: {e}")
            continue
    
    # Print summary
    if downloaded_files:
        print("\n" + "="*70)
        print("📊 DOWNLOADED FILES SUMMARY")
        print("="*70)
        
        total_size = 0
        for name, info in downloaded_files.items():
            size_mb = info['file'].stat().st_size / (1024*1024)
            total_size += info['file'].stat().st_size
            print(f"\n{name.upper()}:")
            print(f"   File: {info['file'].name}")
            print(f"   Records: {info['records']:,}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Columns: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")
        
        print("-"*70)
        print(f"Total Size: {total_size / (1024*1024):.1f} MB")
        print(f"Location: {output_dir}")
        print("="*70)
        
        # Cache cleanup info
        print("\n🧹 To clean up cache and save space:")
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        print(f"   rm -rf {cache_dir}/wzehui___yelp-multimodal-recommendation")
        
        return True
    else:
        print("\n❌ No files were downloaded successfully")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Yelp Multimodal CSV files"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max-records", type=int, default=10000, 
                       help="Maximum records per file")
    
    args = parser.parse_args()
    
    download_yelp_csvs(
        data_dir=args.data_dir,
        max_records=args.max_records
    )
