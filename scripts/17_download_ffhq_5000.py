import json
import os
import urllib.request
import time

# ============================================================
# Download 5,000 FFHQ images
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFHQ_DIR = os.path.join(BASE_DIR, "raw_datasets", "FFHQ")
JSON_PATH = os.path.join(FFHQ_DIR, "ffhq-dataset-v2.json")
OUTPUT_DIR = os.path.join(FFHQ_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("Downloading 5,000 FFHQ Images")
print("=" * 65)

# Load metadata
print("\nLoading FFHQ metadata...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Metadata records: {len(data):,}")

# FFHQ dataset uses 70,000 records.
# Select the first 5,000 IDs.
records = list(data.items())[:5000]

print(f"Selected images: {len(records):,}")
print(f"Output: {OUTPUT_DIR}")

# FFHQ image URL pattern
BASE_URL = "https://github.com/NVlabs/ffhq-dataset/raw/master/thumbnails128x128/"

success = 0
failed = 0

for index, (key, metadata) in enumerate(records, start=1):

    # Metadata keys are normally 00000 ... 69999
    image_id = str(key).zfill(5)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{image_id}.png"
    )

    # Skip existing files
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        success += 1

    else:
        url = f"{BASE_URL}{image_id}.png"

        try:
            urllib.request.urlretrieve(url, output_path)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                success += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1

            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

            print(f"\nFailed: {image_id} -> {e}")

    if index % 100 == 0 or index == len(records):
        print(
            f"Processed {index:,}/{len(records):,} | "
            f"Success: {success:,} | "
            f"Failed: {failed:,}"
        )

print("\n" + "=" * 65)
print("FFHQ DOWNLOAD COMPLETED")
print("=" * 65)
print(f"Success : {success:,}")
print(f"Failed  : {failed:,}")
print(f"Output  : {OUTPUT_DIR}")
print("=" * 65)