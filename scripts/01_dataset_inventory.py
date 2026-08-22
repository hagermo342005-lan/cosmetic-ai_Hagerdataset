from pathlib import Path
from collections import Counter
import csv
from PIL import Image


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASETS = PROJECT_ROOT / "raw_datasets"
OUTPUT_FILE = PROJECT_ROOT / "dataset_inventory.csv"


# =========================
# Dataset information
# =========================

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


# =========================
# Scan one dataset
# =========================

def scan_dataset(dataset_name, dataset_path, source, role):

    files = [
        file
        for file in dataset_path.rglob("*")
        if file.is_file()
    ]

    extension_counter = Counter(
        file.suffix.lower()
        for file in files
    )

    image_files = []

    for file in files:
        try:
            with Image.open(file) as img:
                image_files.append({
                    "file": str(file.relative_to(dataset_path)),
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                })
        except Exception:
            pass
    resolution_counter = Counter(
        f"{item['width']}x{item['height']}"
        for item in image_files
    )

    annotations = [
        file
        for file in files
        if file.suffix.lower() in {
            ".json",
            ".txt",
            ".csv",
            ".xml",
            ".mat",
            ".pts",
            ".xlsx",
        }
    ]

    return {
        "dataset": dataset_name,
        "source": source,
        "role": role,
        "total_files": len(files),
        "image_files": len(image_files),
        "annotation_files": len(annotations),
        "extensions": dict(extension_counter),
        "image_resolutions": dict(resolution_counter),
    
    }


# =========================
# Main
# =========================

def main():

    results = []

    for dataset_name, info in DATASETS.items():

        dataset_path = RAW_DATASETS / dataset_name

        print(f"\nScanning: {dataset_name}")

        if not dataset_path.exists():

            print("  Folder not found.")
            print(f"  Expected: {dataset_path}")

            results.append({
                "dataset": dataset_name,
                "source": info["source"],
                "role": info["role"],
                "total_files": 0,
                "image_files": 0,
                "annotation_files": 0,
                "extensions": {},
                "image_resolutions": {},
            })

            continue

        result = scan_dataset(
            dataset_name,
            dataset_path,
            info["source"],
            info["role"],
        )

        results.append(result)

        print(f"  Total files: {result['total_files']}")
        print(f"  Image files: {result['image_files']}")
        print(f"  Annotation files: {result['annotation_files']}")


    # =========================
    # Save CSV
    # =========================

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "source",
                "role",
                "total_files",
                "image_files",
                "annotation_files",
                "extensions",
                "image_resolutions",
            ],
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "dataset": result["dataset"],
                "source": result["source"],
                "role": result["role"],
                "total_files": result["total_files"],
                "image_files": result["image_files"],
                "annotation_files": result["annotation_files"],
                "extensions": str(result["extensions"]),
                "image_resolutions": str(
                    result["image_resolutions"]
                ),
            })


    print("\n==============================")
    print("Dataset inventory completed.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("==============================")


if __name__ == "__main__":
    main()