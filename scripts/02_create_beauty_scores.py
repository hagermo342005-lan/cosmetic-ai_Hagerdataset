import pandas as pd
from pathlib import Path

# ==============================
# Paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "raw_datasets"
    / "SCUT-FBP5500"
    / "SCUT-FBP5500_v2"
    / "All_Ratings.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "processed" / "beauty"
OUTPUT_FILE = OUTPUT_DIR / "beauty_scores.csv"


# ==============================
# Load ratings
# ==============================

print("Loading SCUT-FBP5500 ratings...")

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="ALL"
)

print(f"Raw rating rows: {len(df):,}")


# ==============================
# Clean data
# ==============================

df = df[["Rater", "Filename", "Rating"]].copy()

df["Filename"] = df["Filename"].astype(str)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Remove invalid ratings
df = df.dropna(subset=["Filename", "Rating"])


# ==============================
# Aggregate ratings
# ==============================

beauty = (
    df.groupby("Filename")
    .agg(
        Mean_Beauty_Score=("Rating", "mean"),
        Std_Beauty_Score=("Rating", "std"),
        Min_Beauty_Score=("Rating", "min"),
        Max_Beauty_Score=("Rating", "max"),
        Num_Raters=("Rating", "count"),
    )
    .reset_index()
)


# ==============================
# Round scores
# ==============================

beauty["Mean_Beauty_Score"] = beauty["Mean_Beauty_Score"].round(4)
beauty["Std_Beauty_Score"] = beauty["Std_Beauty_Score"].round(4)


# ==============================
# Sort
# ==============================

beauty = beauty.sort_values("Filename").reset_index(drop=True)


# ==============================
# Save
# ==============================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

beauty.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ==============================
# Report
# ==============================

print()
print("=" * 50)
print("Beauty Score Dataset Created")
print("=" * 50)

print(f"Images with scores: {len(beauty):,}")
print(f"Output: {OUTPUT_FILE}")

print()
print("Columns:")
print(beauty.columns.tolist())

print()
print("First 10 records:")
print(beauty.head(10).to_string(index=False))

print()
print("Mean score statistics:")
print(beauty["Mean_Beauty_Score"].describe())

print()
print("Raters per image:")
print(beauty["Num_Raters"].value_counts().sort_index())