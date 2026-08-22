import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "raw_datasets"
    / "SCUT-FBP5500"
    / "SCUT-FBP5500_v2"
    / "Images"
)

SCORE_FILE = (
    PROJECT_ROOT
    / "processed"
    / "beauty"
    / "beauty_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "beauty"
    / "samples"
)

OUTPUT_FILE = OUTPUT_DIR / "beauty_samples.png"


# ==========================================
# Load scores
# ==========================================

print("Loading beauty scores...")

df = pd.read_csv(SCORE_FILE)

print(f"Total records: {len(df)}")


# ==========================================
# Select random samples
# ==========================================

random.seed(42)

sample_size = min(12, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42
)


# ==========================================
# Create output directory
# ==========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# Create visualization
# ==========================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(12, 10)
)

axes = axes.flatten()


for ax, (_, row) in zip(axes, sample.iterrows()):

    filename = row["Filename"]

    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        ax.text(
            0.5,
            0.5,
            f"Image not found:\n{filename}",
            ha="center",
            va="center"
        )
        ax.axis("off")
        continue

    image = plt.imread(image_path)

    ax.imshow(image)

    ax.set_title(
        f"{filename}\n"
        f"Beauty Score: {row['Mean_Beauty_Score']:.2f}\n"
        f"Raters: {int(row['Num_Raters'])}"
    )

    ax.axis("off")


# Hide unused axes

for ax in axes[sample_size:]:
    ax.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# Final report
# ==========================================

print()
print("=" * 50)
print("Visualization Created")
print("=" * 50)

print(f"Samples: {sample_size}")
print(f"Output: {OUTPUT_FILE}")