from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# Paths
# ==========================================

FACES_DIR = Path("./processed/faces/CelebAMask-HQ")
MASKS_DIR = Path("./processed/masks")

NUM_IMAGES = 9


# ==========================================
# Find face images
# ==========================================

face_files = sorted(FACES_DIR.glob("*.jpg"))

print("=" * 60)
print("CELEBAMASK-HQ FACE + MASK VISUALIZATION")
print("=" * 60)

print(f"Faces found: {len(face_files)}")


if len(face_files) == 0:
    print("No face images found.")
    print(f"Checked: {FACES_DIR}")
    exit()


# ==========================================
# Find matching masks
# ==========================================

samples = []

for face_path in face_files:

    image_id = face_path.stem

    possible_masks = [
        MASKS_DIR / f"{image_id}.png",
        MASKS_DIR / f"{image_id}.jpg",
    ]

    mask_path = None

    for p in possible_masks:
        if p.exists():
            mask_path = p
            break

    if mask_path is not None:
        samples.append((face_path, mask_path))

    if len(samples) >= NUM_IMAGES:
        break


print(f"Matching face/mask pairs: {len(samples)}")


# ==========================================
# Visualization
# ==========================================

if len(samples) == 0:
    print()
    print("No matching masks were found.")
    print(f"Faces: {FACES_DIR}")
    print(f"Masks: {MASKS_DIR}")
    exit()


fig, axes = plt.subplots(
    len(samples),
    2,
    figsize=(10, 5 * len(samples))
)

if len(samples) == 1:
    axes = [axes]


for row, (face_path, mask_path) in enumerate(samples):

    face = Image.open(face_path).convert("RGB")
    mask = Image.open(mask_path)

    axes[row][0].imshow(face)
    axes[row][0].set_title(
        f"{face_path.stem} - Face"
    )
    axes[row][0].axis("off")

    axes[row][1].imshow(mask, cmap="gray")
    axes[row][1].set_title(
        f"{face_path.stem} - Mask"
    )
    axes[row][1].axis("off")


plt.tight_layout()


# ==========================================
# Save preview
# ==========================================

output = Path(
    "./processed/celeba_hq_faces_masks_preview.jpg"
)

plt.savefig(
    output,
    dpi=150,
    bbox_inches="tight"
)

print()
print(f"Preview saved to:")
print(output)

plt.show()