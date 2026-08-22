from pathlib import Path
from PIL import Image
import csv
import time


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = ROOT / "raw_datasets"
OUTPUT_ROOT = ROOT / "processed" / "images_512"

TARGET_SIZE = (512, 512)

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


# ============================================================
# SETUP
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

report_path = ROOT / "processed" / "resize_512_report.csv"

report_rows = []

start_time = time.time()


# ============================================================
# FUNCTIONS
# ============================================================

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def resize_image(input_path: Path, output_path: Path):
    """
    Resize image to exactly 512x512.
    Raw image is never modified.
    """

    with Image.open(input_path) as img:

        # Convert unsupported modes safely
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        # Resize exactly to 512x512
        resized = img.resize(
            TARGET_SIZE,
            Image.Resampling.LANCZOS
        )

        # Preserve alpha for PNG/RGBA
        if input_path.suffix.lower() == ".png":
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )
            resized.save(
                output_path,
                format="PNG"
            )

        else:
            # Save JPEG outputs as JPEG
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if resized.mode == "RGBA":
                resized = resized.convert("RGB")

            elif resized.mode == "L":
                resized = resized.convert("RGB")

            resized.save(
                output_path,
                format="JPEG",
                quality=95
            )


# ============================================================
# PROCESS DATASETS
# ============================================================

print("=" * 70)
print("DATASET RESIZE → 512x512")
print("=" * 70)

print(f"Raw root    : {RAW_ROOT}")
print(f"Output root : {OUTPUT_ROOT}")
print(f"Target size : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

for dataset_name in DATASETS:

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    input_root = RAW_ROOT / dataset_name
    output_root = OUTPUT_ROOT / dataset_name

    if not input_root.exists():

        print(f"ERROR: Dataset not found: {input_root}")

        report_rows.append({
            "dataset": dataset_name,
            "input_images": 0,
            "processed": 0,
            "failed": 0,
            "output_images": 0,
        })

        continue

    image_files = [
        p for p in input_root.rglob("*")
        if p.is_file() and is_image(p)
    ]

    total = len(image_files)
    processed = 0
    failed = 0

    print(f"Images found: {total}")

    for index, input_path in enumerate(image_files, start=1):

        # Keep same relative folder structure
        relative_path = input_path.relative_to(input_root)

        # Keep PNG as PNG, everything else as JPG
        if input_path.suffix.lower() == ".png":
            output_relative = relative_path.with_suffix(".png")
        else:
            output_relative = relative_path.with_suffix(".jpg")

        output_path = output_root / output_relative

        try:

            resize_image(
                input_path,
                output_path
            )

            processed += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR: {input_path.name}"
                f"\n       {e}"
            )

        # Progress every 100 images
        if index % 100 == 0 or index == total:

            print(
                f"\rProcessed: {index}/{total} "
                f"| Success: {processed} "
                f"| Failed: {failed}",
                end="",
                flush=True
            )

    print()

    print(f"Total images : {total}")
    print(f"Processed    : {processed}")
    print(f"Failed       : {failed}")

    report_rows.append({
        "dataset": dataset_name,
        "input_images": total,
        "processed": processed,
        "failed": failed,
        "output_images": processed,
    })


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "dataset",
        "input_images",
        "processed",
        "failed",
        "output_images",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(report_rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

total_input = sum(
    row["input_images"]
    for row in report_rows
)

total_processed = sum(
    row["processed"]
    for row in report_rows
)

total_failed = sum(
    row["failed"]
    for row in report_rows
)


print("\n" + "=" * 70)
print("RESIZE FINISHED")
print("=" * 70)

print(f"Total input images : {total_input}")
print(f"Total processed    : {total_processed}")
print(f"Total failed       : {total_failed}")

print(f"\nTarget resolution  : 512x512")

print(f"\nOutput:")
print(OUTPUT_ROOT)

print(f"\nReport:")
print(report_path)

print(f"\nTime:")
print(f"{elapsed / 60:.2f} minutes")

print("\nRaw datasets were NOT modified.")
print("=" * 70)