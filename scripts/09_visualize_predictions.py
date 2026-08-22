from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "processed" / "splits" / "test.csv"
LANDMARK_DIR = ROOT / "processed" / "landmarks"
MODEL_PATH = ROOT / "models" / "baseline_landmark_model.joblib"

OUTPUT_DIR = ROOT / "processed" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 12


def load_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = data["landmarks"]

    if len(points) != 86:
        return None

    points = np.array(
        [[p["x"], p["y"]] for p in points],
        dtype=np.float32
    )

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)

    center = (min_xy + max_xy) / 2
    scale = max(max_xy - min_xy)

    if scale == 0:
        return None

    points = (points - center) / scale

    return points.flatten()


print("=" * 60)
print("Visualizing Beauty Predictions")
print("=" * 60)

print("\nLoading test dataset...")
df = pd.read_csv(TEST_CSV)

print(f"Test records: {len(df)}")

print("\nLoading model...")
model = joblib.load(MODEL_PATH)

# Load all valid test samples
records = []

for _, row in df.iterrows():

    landmark_path = LANDMARK_DIR / (
        Path(row["Filename"]).stem + ".json"
    )

    features = load_landmarks(landmark_path)

    if features is None:
        continue

    image_path = ROOT / row["Image_Path"]

    if not image_path.exists():
        continue

    records.append({
        "Filename": row["Filename"],
        "Image_Path": image_path,
        "Actual": float(row["Mean_Beauty_Score"]),
        "Features": features,
    })


print(f"Valid test samples: {len(records)}")

# Predict
X = np.array([r["Features"] for r in records])

predictions = model.predict(X)

for r, prediction in zip(records, predictions):
    r["Predicted"] = float(prediction)
    r["Error"] = abs(r["Actual"] - r["Predicted"])


# ---------------------------------------------------------
# Select samples
# ---------------------------------------------------------

# Pick a mixture of good, medium and difficult predictions
records_sorted = sorted(records, key=lambda x: x["Error"])

if len(records_sorted) >= NUM_SAMPLES:

    best = records_sorted[:4]

    middle_start = len(records_sorted) // 2 - 2
    middle = records_sorted[middle_start:middle_start + 4]

    worst = records_sorted[-4:]

    selected = best + middle + worst

else:
    selected = records_sorted


# ---------------------------------------------------------
# Create image grid
# ---------------------------------------------------------

print("\nCreating prediction visualization...")

fig, axes = plt.subplots(
    3,
    4,
    figsize=(14, 12)
)

axes = axes.flatten()

for i, ax in enumerate(axes):

    if i >= len(selected):
        ax.axis("off")
        continue

    r = selected[i]

    try:
        image = Image.open(r["Image_Path"]).convert("RGB")

        ax.imshow(image)

        ax.set_title(
            f'{r["Filename"]}\n'
            f'Actual: {r["Actual"]:.2f} | '
            f'Pred: {r["Predicted"]:.2f}\n'
            f'Error: {r["Error"]:.2f}',
            fontsize=9
        )

        ax.axis("off")

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error loading image\n{e}",
            ha="center",
            va="center"
        )
        ax.axis("off")


plt.suptitle(
    "Cosmetic AI — Beauty Score Predictions",
    fontsize=18
)

plt.tight_layout()

grid_path = OUTPUT_DIR / "beauty_predictions_grid.png"

plt.savefig(
    grid_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Actual vs Predicted scatter plot
# ---------------------------------------------------------

print("Creating Actual vs Predicted plot...")

actual = np.array([r["Actual"] for r in records])
predicted = np.array([r["Predicted"] for r in records])

plt.figure(figsize=(8, 8))

plt.scatter(
    actual,
    predicted,
    alpha=0.5
)

plt.plot(
    [1, 5],
    [1, 5],
    linestyle="--"
)

plt.xlabel("Actual Beauty Score")
plt.ylabel("Predicted Beauty Score")

plt.title(
    "Actual vs Predicted Beauty Scores"
)

plt.xlim(1, 5)
plt.ylim(1, 5)

plt.grid(True, alpha=0.2)

scatter_path = OUTPUT_DIR / "actual_vs_predicted.png"

plt.savefig(
    scatter_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Save predictions CSV
# ---------------------------------------------------------

prediction_df = pd.DataFrame({
    "Filename": [r["Filename"] for r in records],
    "Actual_Beauty_Score": [r["Actual"] for r in records],
    "Predicted_Beauty_Score": [r["Predicted"] for r in records],
    "Absolute_Error": [r["Error"] for r in records],
})

csv_path = OUTPUT_DIR / "test_predictions.csv"

prediction_df.to_csv(
    csv_path,
    index=False
)


print()
print("=" * 60)
print("Visualization Completed")
print("=" * 60)

print("\nPrediction images:")
print(grid_path)

print("\nActual vs Predicted graph:")
print(scatter_path)

print("\nPrediction CSV:")
print(csv_path)

print("=" * 60)