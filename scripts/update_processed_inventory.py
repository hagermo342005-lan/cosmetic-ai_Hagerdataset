from pathlib import Path
import csv

# ============================================================
# UPDATE PROCESSED DATASET INVENTORY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILES = [
    PROJECT_ROOT / "dataset_inventory.csv",
    PROJECT_ROOT / "processed" / "dataset_inventory.csv",
]

DATASETS = [
    {
        "dataset": "FFHQ",
        "source": "https://github.com/NVlabs/ffhq-dataset",
        "role": "Natural face source for experiments and generation",
        "total_files": 5006,
        "image_files": 5001,
        "annotation_files": 1,
        "extensions": ".json:1; .png:5001; .py:1; .tmp:3",
        "image_resolutions": "512x512:5001",
    },
    {
        "dataset": "FaceSynthetics",
        "source": "https://github.com/microsoft/FaceSynthetics",
        "role": "Landmark pipeline development and testing",
        "total_files": 3001,
        "image_files": 2000,
        "annotation_files": 1000,
        "extensions": ".png:2000; .txt:1000; .zip:1",
        "image_resolutions": "512x512:2000",
    },
    {
        "dataset": "CelebAMask-HQ",
        "source": "https://github.com/switchablenorms/CelebAMask-HQ",
        "role": "Face parsing and mask development/testing",
        "total_files": 402782,
        "image_files": 29299,
        "annotation_files": 4,
        "extensions": ".jpg:30000; .png:372767; .txt:4; [no_extension]:11",
        "image_resolutions": "512x512:29299",
    },
    {
        "dataset": "SCUT-FBP5500",
        "source": "https://github.com/HCIILAB/SCUT-FBP5500-Database-Release",
        "role": "Beauty-score module",
        "total_files": 11017,
        "image_files": 5500,
        "annotation_files": 5516,
        "extensions": ".jpg:5500; .pts:5500; .txt:14; .xlsx:2; .zip:1",
        "image_resolutions": "512x512:5500",
    },
]

FIELDS = [
    "dataset",
    "source",
    "role",
    "total_files",
    "image_files",
    "annotation_files",
    "extensions",
    "image_resolutions",
]

for output_file in OUTPUT_FILES:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(DATASETS)

    print(f"Updated: {output_file}")

print()
print("=" * 60)
print("PROCESSED DATASET INVENTORY UPDATED")
print("=" * 60)
print("All listed processed images are recorded as 512x512.")
print("Raw datasets were NOT modified.")