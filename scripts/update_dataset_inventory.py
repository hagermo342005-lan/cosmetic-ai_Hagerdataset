from pathlib import Path
from collections import Counter
from PIL import Image
import csv
import shutil


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_datasets"
OUTPUT = ROOT / "dataset_inventory.csv"
PROCESSED_OUTPUT = ROOT / "processed" / "dataset_inventory.csv"


DATASETS = {
    "FFHQ": {
        "source": "https://github.com/NVlabs/ffhq-dataset",
        "role": "Natural face source for experiments and generation",
    },
    "FaceSynthetics": {
        "source": "https://github.com/microsoft/FaceSynthetics",
        "role": "Landmark pipeline development and testing",
    },
    "CelebAMask-HQ": {
        "source": "https://github.com/switchablenorms/CelebAMask-HQ",
        "role": "Face parsing and mask development/testing",
    },
    "SCUT-FBP5500": {
        "source": "https://github.com/HCIILAB/SCUT-FBP5500-Database-Release",
        "role": "Beauty-score module",
    },
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


ANNOTATION_EXTENSIONS = {
    ".txt",
    ".pts",
    ".json",
    ".xml",
    ".csv",
    ".xlsx",
    ".mat",
}


rows = []


for dataset, info in DATASETS.items():

    dataset_path = RAW / dataset

    if not dataset_path.exists():
        print(f"[MISSING] {dataset}")
        continue

    files = [
        p for p in dataset_path.rglob("*")
        if p.is_file()
    ]

    total_files = len(files)

    image_files = [
        p for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    annotation_files = [
        p for p in files
        if p.suffix.lower() in ANNOTATION_EXTENSIONS
    ]

    extension_counts = Counter(
        p.suffix.lower() if p.suffix else "[no_extension]"
        for p in files
    )

    resolution_counts = Counter()

    for image_path in image_files:

        try:
            with Image.open(image_path) as img:
                resolution_counts[
                    f"{img.width}x{img.height}"
                ] += 1

        except Exception:
            pass

    extensions_text = "; ".join(
        f"{ext}:{count}"
        for ext, count in sorted(extension_counts.items())
    )

    resolutions_text = "; ".join(
        f"{resolution}:{count}"
        for resolution, count in sorted(resolution_counts.items())
    )

    rows.append({
        "dataset": dataset,
        "source": info["source"],
        "role": info["role"],
        "total_files": total_files,
        "image_files": len(image_files),
        "annotation_files": len(annotation_files),
        "extensions": extensions_text,
        "image_resolutions": resolutions_text,
    })

    print(f"\n{dataset}")
    print(f"  Total files       : {total_files}")
    print(f"  Image files       : {len(image_files)}")
    print(f"  Annotation files  : {len(annotation_files)}")
    print(f"  Extensions        : {extensions_text}")
    print(f"  Resolutions       : {resolutions_text}")


fieldnames = [
    "dataset",
    "source",
    "role",
    "total_files",
    "image_files",
    "annotation_files",
    "extensions",
    "image_resolutions",
]


with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


PROCESSED_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

shutil.copy2(
    OUTPUT,
    PROCESSED_OUTPUT
)


print("\n========================================")
print("DATASET INVENTORY UPDATED")
print("========================================")
print(f"Root     : {OUTPUT}")
print(f"Processed: {PROCESSED_OUTPUT}")