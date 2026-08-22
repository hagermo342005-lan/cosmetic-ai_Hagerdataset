from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

METADATA = ROOT / "processed" / "metadata" / "metadata.csv"
EXCLUSIONS = ROOT / "evaluation" / "dataset_exclusions.csv"
OUTPUT_DIR = ROOT / "processed" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Creating Train / Validation / Test Splits")
print("=" * 60)

# Load metadata
df = pd.read_csv(METADATA)

print(f"Original records: {len(df):,}")

# Remove excluded samples
if EXCLUSIONS.exists():
    exclusions = pd.read_csv(EXCLUSIONS)
    excluded_files = set(exclusions["Filename"].astype(str))
    df = df[~df["Filename"].isin(excluded_files)].copy()

print(f"Valid records: {len(df):,}")

# Stratification key
df["Stratify"] = (
    df["Race"].astype(str)
    + "_"
    + df["Gender"].astype(str)
)

# 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Stratify"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Stratify"]
)

# Remove helper column
for data in (train_df, val_df, test_df):
    data.drop(columns=["Stratify"], inplace=True)

# Save
train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print()
print("=" * 60)
print("Split Results")
print("=" * 60)

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")
print(f"Total      : {len(train_df) + len(val_df) + len(test_df):,}")

print()
print("Train distribution:")
print(train_df.groupby(["Race", "Gender"]).size())

print()
print("Validation distribution:")
print(val_df.groupby(["Race", "Gender"]).size())

print()
print("Test distribution:")
print(test_df.groupby(["Race", "Gender"]).size())

print()
print("Outputs:")
print(train_path)
print(val_path)
print(test_path)

print("=" * 60)
print("Dataset splitting completed.")
print("=" * 60)