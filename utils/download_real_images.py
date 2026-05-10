"""
Fast Yelp image downloader using direct CDN URLs.

The photo_ids in photo_clean.csv are real Yelp photo IDs.
Yelp CDN URL pattern:
  https://s3-media{0-3}.fl.yelpcdn.com/bphoto/{photo_id}/348s.jpg

No search engine required — direct HTTP downloads with 30 workers.
"""
import os
import pandas as pd
import requests
import concurrent.futures
import time
import random
from pathlib import Path

IMG_DIR  = "data/raw/yelp_multimodal_final/images"
PHOTO_CSV = "data/raw/yelp_multimodal_final/photo_clean.csv"

# Yelp CDN mirrors  (rotate to spread load)
CDN_HOSTS = [
    "s3-media0.fl.yelpcdn.com",
    "s3-media1.fl.yelpcdn.com",
    "s3-media2.fl.yelpcdn.com",
    "s3-media3.fl.yelpcdn.com",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.yelp.com/",
    "Accept": "image/webp,image/*,*/*;q=0.8",
}


def is_real_image(path: str) -> bool:
    """Return True if the file exists and is NOT a dummy (>6 KB content variance check)."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    # Dummy placeholder images are tiny grey squares ≤10 KB
    return size > 15_000   # real photos are typically 50–300 KB


def download_one(photo_id: str) -> tuple[str, bool]:
    save_path = os.path.join(IMG_DIR, f"{photo_id}.jpg")

    # Skip if we already have a real image
    if is_real_image(save_path):
        return photo_id, True

    # Try each CDN mirror with the 'o' (original) and '348s' (thumbnail) sizes
    host = random.choice(CDN_HOSTS)
    for size in ("o", "l", "348s"):
        url = f"https://{host}/bphoto/{photo_id}/{size}.jpg"
        try:
            r = requests.get(url, headers=HEADERS, timeout=8, stream=True)
            if r.status_code == 200:
                content = r.content
                # Make sure it's at least 10 KB (a real image)
                if len(content) > 10_000:
                    with open(save_path, "wb") as f:
                        f.write(content)
                    return photo_id, True
        except (requests.RequestException, OSError):
            pass

    return photo_id, False


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    df = pd.read_csv(PHOTO_CSV)
    photo_ids = df["photo_id"].dropna().unique().tolist()
    print(f"Total unique photos: {len(photo_ids)}")

    # Skip those already downloaded as real images
    todo = [pid for pid in photo_ids
            if not is_real_image(os.path.join(IMG_DIR, f"{pid}.jpg"))]
    already_done = len(photo_ids) - len(todo)
    print(f"Already have real images : {already_done}")
    print(f"Need to download         : {len(todo)}")
    print("Downloading with 30 parallel workers from Yelp CDN…\n")

    success = already_done
    failed  = []
    total   = len(photo_ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_one, pid): pid for pid in todo}

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            pid, ok = future.result()
            if ok:
                success += 1
            else:
                failed.append(pid)

            if i % 100 == 0 or i == len(todo):
                print(f"  Progress {i:>4}/{len(todo)}  "
                      f"✓ {success}  ✗ {len(failed)}", flush=True)

    print(f"\n✅ Done. {success}/{total} real images saved to {IMG_DIR}/")
    if failed:
        print(f"⚠️  {len(failed)} failed (CDN blocked or no image exists):")
        for pid in failed[:10]:
            print(f"   {pid}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f}s")
