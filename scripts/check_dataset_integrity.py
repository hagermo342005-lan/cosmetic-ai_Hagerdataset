from pathlib import Path
from PIL import Image
import csv
import time


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_datasets"
REPORT = ROOT / "dataset_integrity_report.csv"

DATASETS = [
    "FFHQ",
    "FaceSynthetics",
    "CelebAMask-HQ",
    "SCUT-FBP5500",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


rows = []
start_time = time.time()


for dataset in DATASETS:

    dataset_path = RAW / dataset

    images = [
        p for p in dataset_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total = len(images)
    valid = 0
    corrupted = 0

    print("\n========================================")
    print(dataset)
    print("========================================")
    print(f"Images to check: {total}")

    for i, image_path in enumerate(images, 1):

        try:
            with Image.open(image_path) as img:
                img.verify()

            valid += 1

        except Exception:
            corrupted += 1

            rows.append({
                "dataset": dataset,
                "file": str(image_path.relative_to(ROOT)),
                "status": "CORRUPTED",
            })

        if i % 1000 == 0 or i == total:
            print(
                f"\rChecked: {i}/{total} | "
                f"Valid: {valid} | "
                f"Corrupted: {corrupted}",
                end=""
            )

    print()

    rows.append({
        "dataset": dataset,
        "file": "",
        "status": (
            f"SUMMARY | total={total} | "
            f"valid={valid} | corrupted={corrupted}"
        ),
    })


with open(REPORT, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "file", "status"]
    )

    writer.writeheader()
    writer.writerows(rows)


elapsed = time.time() - start_time

print("\n========================================")
print("DATASET INTEGRITY CHECK FINISHED")
print("========================================")
print(f"Report: {REPORT}")
print(f"Time: {elapsed / 60:.2f} minutes")