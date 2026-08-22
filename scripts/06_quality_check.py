from pathlib import Path
import json
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = ROOT / "processed" / "metadata" / "metadata.csv"

OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "quality_report.txt"


def main():

    print("=" * 60)
    print("Dataset Quality Control")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_FILE)

    report = []

    # =========================================
    # 1. Basic information
    # =========================================

    report.append("DATASET QUALITY REPORT")
    report.append("=" * 60)

    report.append(f"Total records: {len(df)}")

    # =========================================
    # 2. Missing values
    # =========================================

    missing = df.isnull().sum()

    total_missing = int(missing.sum())

    report.append("\nMissing values:")
    report.append(missing.to_string())
    report.append(f"\nTotal missing values: {total_missing}")

    # =========================================
    # 3. Duplicate filenames
    # =========================================

    duplicates = int(df["Filename"].duplicated().sum())

    report.append("\nDuplicate filenames:")
    report.append(str(duplicates))

    # =========================================
    # 4. Image existence
    # =========================================

    missing_images = int(
        (~df["Image_Exists"]).sum()
    )

    report.append("\nMissing images:")
    report.append(str(missing_images))

    # =========================================
    # 5. Landmark existence
    # =========================================

    missing_landmarks = int(
        (~df["Landmark_Exists"]).sum()
    )

    report.append("\nMissing landmarks:")
    report.append(str(missing_landmarks))

    # =========================================
    # 6. Beauty score validation
    # =========================================

    invalid_scores = df[
        (df["Mean_Beauty_Score"] < 1) |
        (df["Mean_Beauty_Score"] > 5)
    ]

    report.append("\nInvalid beauty scores:")
    report.append(str(len(invalid_scores)))

    # =========================================
    # 7. Number of raters
    # =========================================

    invalid_raters = df[
        df["Num_Raters"] != 60
    ]

    report.append("\nImages without exactly 60 raters:")
    report.append(str(len(invalid_raters)))

    # =========================================
    # 8. Image integrity
    # =========================================

    bad_images = []

    print("\nChecking image files...")

    for i, row in enumerate(df.itertuples(), start=1):

        image_path = ROOT / row.Image_Path

        try:
            with Image.open(image_path) as img:
                img.verify()

        except Exception:
            bad_images.append(row.Filename)

        if i % 500 == 0:
            print(f"Checked {i}/{len(df)} images")

    report.append("\nCorrupted/unreadable images:")
    report.append(str(len(bad_images)))

    if bad_images:
        report.append("\nBad image filenames:")
        report.extend(bad_images)

    # =========================================
    # 9. Landmark validation
    # =========================================

    bad_landmarks = []

    print("\nChecking landmark files...")

    for i, row in enumerate(df.itertuples(), start=1):

        landmark_path = ROOT / row.Landmark_Path

        try:

            with open(
                landmark_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            points = data.get("landmarks", [])

            if len(points) != 86:
                bad_landmarks.append(
                    f"{row.Filename}: {len(points)} points"
                )

        except Exception:
            bad_landmarks.append(
                f"{row.Filename}: invalid JSON"
            )

        if i % 500 == 0:
            print(f"Checked {i}/{len(df)} landmarks")

    report.append("\nInvalid landmark files:")
    report.append(str(len(bad_landmarks)))

    if bad_landmarks:
        report.append("\nBad landmark files:")
        report.extend(bad_landmarks)

    # =========================================
    # 10. Final status
    # =========================================

    problems = (
        total_missing
        + duplicates
        + missing_images
        + missing_landmarks
        + len(invalid_scores)
        + len(invalid_raters)
        + len(bad_images)
        + len(bad_landmarks)
    )

    report.append("\n" + "=" * 60)

    if problems == 0:
        report.append("FINAL STATUS: PASS")
    else:
        report.append("FINAL STATUS: REVIEW REQUIRED")

    report.append(f"Total detected problems: {problems}")

    report.append("=" * 60)

    # =========================================
    # Save report
    # =========================================

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # =========================================
    # Print report
    # =========================================

    print("\n".join(report))

    print("\nReport saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()