from pathlib import Path
import json
import random

import cv2
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "raw_datasets"
    / "SCUT-FBP5500"
    / "SCUT-FBP5500_v2"
    / "Images"
)

LANDMARK_DIR = (
    PROJECT_ROOT
    / "processed"
    / "landmarks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "landmarks"
    / "samples"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Settings
# =========================

NUM_SAMPLES = 12

random.seed(42)


# =========================
# Load samples
# =========================

json_files = sorted(LANDMARK_DIR.glob("*.json"))

if not json_files:
    raise RuntimeError("No landmark JSON files found.")

sample_files = random.sample(
    json_files,
    min(NUM_SAMPLES, len(json_files))
)


# =========================
# Create visualization
# =========================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)

axes = axes.flatten()


for ax, json_path in zip(axes, sample_files):

    # -------------------------
    # Load landmark JSON
    # -------------------------

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data["image"]
    landmarks = data["landmarks"]

    image_path = IMAGE_DIR / image_name

    # -------------------------
    # Load image
    # -------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        ax.set_title(f"{image_name}\nImage not found")
        ax.axis("off")
        continue

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -------------------------
    # Display image
    # -------------------------

    ax.imshow(image)

    # -------------------------
    # Draw landmarks
    # -------------------------

    x = [point["x"] for point in landmarks]
    y = [point["y"] for point in landmarks]

    ax.scatter(
        x,
        y,
        s=8
    )

    ax.set_title(
        f"{image_name}\n{len(landmarks)} landmarks"
    )

    ax.axis("off")


# =========================
# Save result
# =========================

plt.tight_layout()

output_path = OUTPUT_DIR / "scut_landmarks_samples.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("=" * 60)
print("Landmark Visualization Created")
print("=" * 60)

print(f"Samples: {len(sample_files)}")
print(f"Output:  {output_path}")

print("=" * 60)