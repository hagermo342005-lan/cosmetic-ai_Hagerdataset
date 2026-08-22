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

TARGET = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FFHQ - Download 5,000 Valid Images")
print("=" * 70)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Metadata records: {len(data):,}")
print(f"Target images   : {TARGET:,}")
print(f"Output          : {OUTPUT_DIR}")

success = 0
failed = 0
skipped = 0

for i in range(len(data)):

    if success >= TARGET:
        break

    key = str(i)

    if key not in data:
        failed += 1
        continue

    item = data[key]["image"]

    expected_size = item["file_size"]
    url = item["file_url"]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{i:05d}.png"
    )

    # Already downloaded correctly
    if os.path.exists(output_path):

        current_size = os.path.getsize(output_path)

        if current_size == expected_size:
            success += 1
            skipped += 1

            if success % 100 == 0:
                print(
                    f"Progress: {success:,}/{TARGET:,} | "
                    f"Failed: {failed:,}"
                )

            continue

        else:
            try:
                os.remove(output_path)
            except:
                pass

    print(
        f"[{success + 1:,}/{TARGET:,}] "
        f"Downloading {i:05d}...",
        end=" ",
        flush=True
    )

    downloaded = False

    for attempt in range(3):

        try:

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
                raise IOError(
                    f"Size mismatch: "
                    f"{actual_size} != {expected_size}"
                )

            temp_path = output_path + ".tmp"

            with open(temp_path, "wb") as f:
                f.write(content)

            os.replace(temp_path, output_path)

            downloaded = True
            success += 1

            print(
                f"SUCCESS ({actual_size:,} bytes)"
            )

            break

        except Exception as e:

            if attempt < 2:
                time.sleep(2)

            else:
                print(
                    f"FAILED: {type(e).__name__}: {e}"
                )

    if not downloaded:
        failed += 1

    # Small delay to avoid hammering Google Drive
    time.sleep(0.3)

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETED")
print("=" * 70)

print(f"Successful images : {success:,}")
print(f"Already existed   : {skipped:,}")
print(f"Failed attempts   : {failed:,}")

print(f"\nOutput:")
print(OUTPUT_DIR)

print("=" * 70)