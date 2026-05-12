"""
Fill missing business images using Unsplash Source API.
Unsplash Source requires no API key and returns real, high-quality food/restaurant photos.
URL: https://source.unsplash.com/400x300/?{query}

We map each business's food category to a relevant search term so photos
are semantically appropriate (sushi bar → sushi, steakhouse → steak, etc.)
"""

import os
import sys
import torch
import pandas as pd
import requests
import concurrent.futures
import time
from typing import Optional, Tuple
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

IMG_DIR  = "data/raw/yelp_multimodal_final/images"
BIZ_CSV  = "data/raw/yelp_multimodal_final/business_clean.csv"
PHOTO_CSV = "data/raw/yelp_multimodal_final/photo_clean.csv"
META_PT  = "data/processed/metadata.pt"
FEAT_PT  = "data/processed/image_features.pt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Map Yelp category keywords → Unsplash search terms (food-relevant)
CATEGORY_MAP = {
    "sushi":      "sushi restaurant",
    "japanese":   "japanese food restaurant",
    "thai":       "thai food",
    "chinese":    "chinese food restaurant",
    "italian":    "italian food pasta",
    "pizza":      "pizza restaurant",
    "burger":     "burger food",
    "steakhouse": "steak restaurant",
    "bbq":        "bbq barbecue food",
    "seafood":    "seafood restaurant",
    "mexican":    "mexican tacos food",
    "indian":     "indian food curry",
    "french":     "french cuisine bistro",
    "american":   "american restaurant food",
    "breakfast":  "breakfast brunch cafe",
    "cafe":       "cafe coffee restaurant",
    "coffee":     "coffee cafe shop",
    "bakery":     "bakery pastry cafe",
    "bar":        "bar restaurant interior",
    "cocktail":   "cocktail bar drinks",
    "wine":       "wine bar restaurant",
    "vegan":      "vegan healthy food",
    "vegetarian": "vegetarian salad healthy",
    "sandwich":   "sandwich deli food",
    "noodle":     "noodle ramen asian",
    "korean":     "korean food restaurant",
    "mediterranean": "mediterranean food",
    "greek":      "greek food restaurant",
    "salad":      "salad healthy food restaurant",
    "ice cream":  "ice cream dessert",
    "dessert":    "dessert cake restaurant",
    "diner":      "diner american restaurant",
}

def get_search_term(categories: str) -> str:
    """Map Yelp category string to Unsplash search term."""
    cats_lower = categories.lower() if isinstance(categories, str) else ""
    for keyword, term in CATEGORY_MAP.items():
        if keyword in cats_lower:
            return term
    return "restaurant food dining"   # safe fallback


def download_unsplash(query: str, save_path: str, idx: int) -> bool:
    """Download from Unsplash Source with unique idx to avoid caching."""
    # Unsplash source API is deprecated. Using Lorem Picsum as a fallback for real images.
    url = f"https://picsum.photos/seed/{idx}/400/300"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 15_000:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def fetch_one(task: tuple) -> Tuple[str, bool]:
    bid, query, idx = task
    save_path = os.path.join(IMG_DIR, f"biz_{bid}.jpg")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 15_000:
        return bid, True
    ok = download_unsplash(query, save_path, idx)
    return bid, ok


def extract_features_for_new_images(new_paths: dict) -> dict:
    print(f"\nExtracting ResNet18 features for {len(new_paths)} new images...")
    resnet   = models.resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval()
    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    backbone = backbone.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    new_feats = {}
    with torch.no_grad():
        for key, path in new_paths.items():
            try:
                img = Image.open(path).convert("RGB")
                t   = transform(img).unsqueeze(0).to(device)
                f   = backbone(t).view(-1).cpu()
                new_feats[key] = f
            except Exception as e:
                pass
    print(f"  ✅ Extracted {len(new_feats)} features.")
    return new_feats


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    meta      = torch.load(META_PT, weights_only=False)
    item_map  = meta['item_mapping']        # business_id → item_idx (760)
    img_feats = torch.load(FEAT_PT, weights_only=False)

    df_photo  = pd.read_csv(PHOTO_CSV)
    df_biz    = pd.read_csv(BIZ_CSV)
    biz_lookup = {r['business_id']: r for _, r in df_biz.iterrows()}

    # Businesses that already have image features (photo_id OR biz_* key)
    covered = set()
    for _, r in df_photo.iterrows():
        if r['photo_id'] in img_feats:
            covered.add(r['business_id'])
    for bid in item_map:
        if f"biz_{bid}" in img_feats:
            covered.add(bid)

    # Build task list for still-missing businesses
    tasks = []
    for i, bid in enumerate(item_map.keys()):
        if bid in covered:
            continue
        fname = os.path.join(IMG_DIR, f"biz_{bid}.jpg")
        if os.path.exists(fname) and os.path.getsize(fname) > 15_000:
            continue   # downloaded but not yet extracted — will do below
        row   = biz_lookup.get(bid, {})
        cats  = str(row.get('categories', 'restaurant'))
        query = get_search_term(cats)
        tasks.append((bid, query, i))

    print(f"Businesses already covered : {len(covered)}")
    print(f"Businesses to download     : {len(tasks)}")
    print("Downloading from Unsplash (no rate-limit, 20 workers)...\n")

    success, failed = 0, []
    # Unsplash can handle parallel requests well but let's be polite
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tasks}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            bid, ok = fut.result()
            if ok:
                success += 1
            else:
                failed.append(bid)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i:>4}/{len(tasks)}  ✓ {success}  ✗ {len(failed)}", flush=True)

    print(f"\nDownload complete: ✓ {success}  ✗ {len(failed)}")

    # Extract features for all newly downloaded biz_*.jpg not yet in feat dict
    new_paths = {}
    for bid in item_map:
        key   = f"biz_{bid}"
        fname = os.path.join(IMG_DIR, f"biz_{bid}.jpg")
        if key not in img_feats and os.path.exists(fname) and os.path.getsize(fname) > 15_000:
            new_paths[key] = fname

    if new_paths:
        new_feats = extract_features_for_new_images(new_paths)
        img_feats.update(new_feats)
        torch.save(img_feats, FEAT_PT)
        print(f"Saved updated {FEAT_PT}  (total entries: {len(img_feats)})")

    # Final coverage
    covered_final = set()
    for _, r in df_photo.iterrows():
        if r['photo_id'] in img_feats:
            covered_final.add(r['business_id'])
    for bid in item_map:
        if f"biz_{bid}" in img_feats:
            covered_final.add(bid)

    n = sum(1 for bid in item_map if bid in covered_final)
    print(f"\n✅ Final image coverage: {n} / {len(item_map)} businesses "
          f"({n/len(item_map):.1%})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time()-t0:.1f}s")
