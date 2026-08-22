from pathlib import Path
import pandas as pd


# ============================================
# Paths
# ============================================

ROOT = Path(__file__).resolve().parent.parent

BEAUTY_FILE = (
    ROOT
    / "processed"
    / "beauty"
    / "beauty_scores.csv"
)

LANDMARK_DIR = (
    ROOT
    / "processed"
    / "landmarks"
)

IMAGE_DIR = (
    ROOT
    / "raw_datasets"
    / "SCUT-FBP5500"
    / "SCUT-FBP5500_v2"
    / "Images"
)

OUTPUT_DIR = (
    ROOT
    / "processed"
    / "metadata"
)

OUTPUT_FILE = OUTPUT_DIR / "metadata.csv"


# ============================================
# Main
# ============================================

def main():

    print("=" * 60)
    print("Creating Unified Metadata")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Load beauty scores
    # ----------------------------------------

    print("\nLoading beauty scores...")

    beauty_df = pd.read_csv(BEAUTY_FILE)

    print(f"Beauty records: {len(beauty_df):,}")

    # ----------------------------------------
    # Create metadata
    # ----------------------------------------

    records = []

    for _, row in beauty_df.iterrows():

        filename = row["Filename"]

        # ------------------------------------
        # Determine gender / race
        # ------------------------------------

        prefix = filename[:2].upper()

        if prefix == "AF":
            race = "Asian"
            gender = "Female"

        elif prefix == "AM":
            race = "Asian"
            gender = "Male"

        elif prefix == "CF":
            race = "Caucasian"
            gender = "Female"

        elif prefix == "CM":
            race = "Caucasian"
            gender = "Male"

        else:
            race = "Unknown"
            gender = "Unknown"

        # ------------------------------------
        # Paths
        # ------------------------------------

        image_path = IMAGE_DIR / filename

        landmark_filename = Path(filename).stem + ".json"

        landmark_path = LANDMARK_DIR / landmark_filename

        # ------------------------------------
        # Check files
        # ------------------------------------

        image_exists = image_path.exists()

        landmark_exists = landmark_path.exists()

        # ------------------------------------
        # Create record
        # ------------------------------------

        records.append({
            "Filename": filename,

            "Image_Path": str(
                image_path.relative_to(ROOT)
            ),

            "Landmark_Path": str(
                landmark_path.relative_to(ROOT)
            ),

            "Race": race,

            "Gender": gender,

            "Mean_Beauty_Score": row["Mean_Beauty_Score"],

            "Std_Beauty_Score": row["Std_Beauty_Score"],

            "Min_Beauty_Score": row["Min_Beauty_Score"],

            "Max_Beauty_Score": row["Max_Beauty_Score"],

            "Num_Raters": row["Num_Raters"],

            "Image_Exists": image_exists,

            "Landmark_Exists": landmark_exists,
        })

    # ----------------------------------------
    # Create DataFrame
    # ----------------------------------------

    metadata_df = pd.DataFrame(records)

    # ----------------------------------------
    # Save
    # ----------------------------------------

    metadata_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ----------------------------------------
    # Statistics
    # ----------------------------------------

    print("\n" + "=" * 60)
    print("Metadata Created")
    print("=" * 60)

    print(f"\nTotal records: {len(metadata_df):,}")

    print(
        f"Images found: "
        f"{metadata_df['Image_Exists'].sum():,}"
    )

    print(
        f"Landmarks found: "
        f"{metadata_df['Landmark_Exists'].sum():,}"
    )

    print("\nRace distribution:")

    print(
        metadata_df["Race"]
        .value_counts()
        .to_string()
    )

    print("\nGender distribution:")

    print(
        metadata_df["Gender"]
        .value_counts()
        .to_string()
    )

    print("\nOutput:")

    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()