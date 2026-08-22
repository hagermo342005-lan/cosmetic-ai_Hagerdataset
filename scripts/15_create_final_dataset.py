from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = ROOT / "processed" / "splits"
LANDMARKS_DIR = ROOT / "processed" / "landmarks"
OUTPUT_DIR = ROOT / "processed" / "final_dataset"

SPLITS = ["train", "validation", "test"]

print("=" * 70)
print("Creating Final Complete Dataset")
print("=" * 70)

total_images = 0
total_landmarks = 0
total_records = 0

for split in SPLITS:

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    csv_path = SPLITS_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)

    split_dir = OUTPUT_DIR / split
    images_dir = split_dir / "images"
    landmarks_dir = split_dir / "landmarks"

    images_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    final_rows = []

    for _, row in df.iterrows():

        filename = row["Filename"]

        source_image = ROOT / row["Image_Path"]
        source_landmark = ROOT / row["Landmark_Path"]

        destination_image = images_dir / filename
        destination_landmark = landmarks_dir / f"{Path(filename).stem}.json"

        if not source_image.exists():
            print(f"WARNING: Missing image: {filename}")
            continue

        if not source_landmark.exists():
            print(f"WARNING: Missing landmark: {filename}")
            continue

        shutil.copy2(source_image, destination_image)
        shutil.copy2(source_landmark, destination_landmark)

        final_rows.append({
            "Filename": filename,
            "Image_Path": f"images/{filename}",
            "Landmark_Path": f"landmarks/{Path(filename).stem}.json",
            "Race": row["Race"],
            "Gender": row["Gender"],
            "Mean_Beauty_Score": row["Mean_Beauty_Score"],
            "Std_Beauty_Score": row["Std_Beauty_Score"],
            "Min_Beauty_Score": row["Min_Beauty_Score"],
            "Max_Beauty_Score": row["Max_Beauty_Score"],
            "Num_Raters": row["Num_Raters"],
        })

    final_df = pd.DataFrame(final_rows)

    labels_path = split_dir / "labels.csv"
    final_df.to_csv(labels_path, index=False)

    image_count = len(list(images_dir.glob("*.jpg")))
    landmark_count = len(list(landmarks_dir.glob("*.json")))

    print(f"Records       : {len(final_df)}")
    print(f"Images        : {image_count}")
    print(f"Landmarks     : {landmark_count}")
    print(f"Labels        : {labels_path}")

    total_records += len(final_df)
    total_images += image_count
    total_landmarks += landmark_count

print("\n" + "=" * 70)
print("FINAL DATASET CREATED")
print("=" * 70)

print(f"Total records   : {total_records}")
print(f"Total images    : {total_images}")
print(f"Total landmarks : {total_landmarks}")

print("\nOutput:")
print(OUTPUT_DIR)

print("=" * 70)