from pathlib import Path
import json
import cv2

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "processed" / "final_dataset"
OUTPUT_DIR = ROOT / "processed" / "final_dataset_visualized"

SPLITS = ["train", "validation", "test"]

print("=" * 70)
print("Creating Landmark Visualization for Full Dataset")
print("=" * 70)

total_processed = 0
total_failed = 0

for split in SPLITS:

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    images_dir = DATASET_DIR / split / "images"
    landmarks_dir = DATASET_DIR / split / "landmarks"

    output_dir = OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))

    print(f"Images found: {len(image_files)}")

    processed = 0
    failed = 0

    for i, image_path in enumerate(image_files, 1):

        landmark_path = landmarks_dir / f"{image_path.stem}.json"

        if not landmark_path.exists():
            print(f"WARNING: Missing landmark: {image_path.name}")
            failed += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Cannot read image: {image_path.name}")
            failed += 1
            continue

        try:
            with open(landmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            landmarks = data.get("landmarks", [])

            if len(landmarks) != 86:
                print(
                    f"WARNING: {image_path.name} "
                    f"has {len(landmarks)} landmarks instead of 86"
                )
                failed += 1
                continue

            # Draw all 86 landmarks
            for point in landmarks:

                x = int(round(point["x"]))
                y = int(round(point["y"]))

                cv2.circle(
                    image,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            # Draw landmark number
            for idx, point in enumerate(landmarks):

                x = int(round(point["x"]))
                y = int(round(point["y"]))

                cv2.putText(
                    image,
                    str(idx + 1),
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            output_path = output_dir / image_path.name

            cv2.imwrite(str(output_path), image)

            processed += 1
            total_processed += 1

            if i % 500 == 0 or i == len(image_files):
                print(
                    f"Processed {i}/{len(image_files)} "
                    f"| Success: {processed} "
                    f"| Failed: {failed}"
                )

        except Exception as e:
            print(f"ERROR: {image_path.name} -> {e}")
            failed += 1
            total_failed += 1

    print(f"\n{split.upper()} completed:")
    print(f"Processed: {processed}")
    print(f"Failed:    {failed}")

print("\n" + "=" * 70)
print("FULL LANDMARK VISUALIZATION COMPLETED")
print("=" * 70)

print(f"Total processed: {total_processed}")
print(f"Total failed:    {total_failed}")

print("\nOutput:")
print(OUTPUT_DIR)

print("=" * 70)