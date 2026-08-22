import json
import os
import time
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(
    BASE_DIR,
    "raw_datasets",
    "FFHQ",
    "ffhq-dataset-v2.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "raw_datasets",
    "FFHQ",
    "images"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FFHQ Download Test")
print("=" * 70)

print("\nLoading metadata...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Metadata records: {len(data):,}")

success = 0
failed = 0

# Test first 10 images
for i in range(10):

    key = str(i)

    if key not in data:
        print(f"Missing metadata: {key}")
        failed += 1
        continue

    item = data[key]["image"]

    url = item["file_url"]
    expected_size = item["file_size"]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{i:05d}.png"
    )

    print(f"\nDownloading {i:05d}...")
    print(f"URL: {url}")

    try:

        # Skip if already correctly downloaded
        if os.path.exists(output_path):

            current_size = os.path.getsize(output_path)

            if current_size == expected_size:
                print("Already exists and size is correct.")
                success += 1
                continue

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=60
        ) as response:

            content = response.read()

        actual_size = len(content)

        if actual_size != expected_size:

            print(
                f"FAILED: size mismatch "
                f"(expected {expected_size}, got {actual_size})"
            )

            failed += 1
            continue

        with open(output_path, "wb") as f:
            f.write(content)

        print(
            f"SUCCESS: {actual_size:,} bytes"
        )

        success += 1

        time.sleep(0.5)

    except urllib.error.HTTPError as e:

        print(
            f"HTTP ERROR {e.code}: {e.reason}"
        )

        failed += 1

    except Exception as e:

        print(
            f"ERROR: {type(e).__name__}: {e}"
        )

        failed += 1


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)

print(f"Success : {success}")
print(f"Failed  : {failed}")
print(f"Output  : {OUTPUT_DIR}")
print("=" * 70)