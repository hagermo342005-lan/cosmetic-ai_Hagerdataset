from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = ROOT / "processed" / "splits"
OUTPUT_DIR = ROOT / "processed" / "dataset"

SPLITS = {
    "train": SPLITS_DIR / "train.csv",
    "validation": SPLITS_DIR / "validation.csv",
    "test": SPLITS_DIR / "test.csv",
}

print("=" * 60)
print("Exporting Physical Dataset")
print("=" * 60)

total_copied = 0
total_missing = 0

for split_name, csv_path in SPLITS.items():

    print(f"\nProcessing: {split_name}")

    df = pd.read_csv(csv_path)

    output_split = OUTPUT_DIR / split_name
    output_split.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for _, row in df.iterrows():

        source = ROOT / row["Image_Path"]
        destination = output_split / row["Filename"]

        if not source.exists():
            print(f"WARNING: Missing image: {source}")
            missing += 1
            continue

        shutil.copy2(source, destination)
        copied += 1

    print(f"Expected : {len(df)}")
    print(f"Copied   : {copied}")
    print(f"Missing  : {missing}")

    total_copied += copied
    total_missing += missing

print("\n" + "=" * 60)
print("Export Completed")
print("=" * 60)

print(f"Total copied : {total_copied}")
print(f"Total missing: {total_missing}")

print("\nOutput:")
print(OUTPUT_DIR)

print("=" * 60)