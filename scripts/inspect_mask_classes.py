from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Files
# ==========================================

face_path = Path("./processed/faces/CelebAMask-HQ/0.jpg")
mask_path = Path("./processed/masks/00000_mask.png")

# ==========================================
# Load
# ==========================================

face = np.array(Image.open(face_path).convert("RGB"))
mask = np.array(Image.open(mask_path))

# ==========================================
# Print classes
# ==========================================

classes, counts = np.unique(mask, return_counts=True)

print("=" * 60)
print("MASK CLASS INSPECTION")
print("=" * 60)

for cls, count in zip(classes, counts):
    percentage = count / mask.size * 100
    print(
        f"Class {cls:>2}: "
        f"{count:>8} pixels "
        f"({percentage:>6.2f}%)"
    )

# ==========================================
# Display
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].imshow(face)
axes[0].set_title("Original Face")
axes[0].axis("off")

# Raw class mask
axes[1].imshow(mask)
axes[1].set_title("Multi-Class Mask")
axes[1].axis("off")

# Overlay
axes[2].imshow(face)
axes[2].imshow(mask, alpha=0.45)
axes[2].set_title("Face + Mask Overlay")
axes[2].axis("off")

plt.tight_layout()

output = Path("./processed/mask_class_inspection_00000.jpg")

plt.savefig(
    output,
    dpi=150,
    bbox_inches="tight"
)

print()
print("Preview saved to:")
print(output)

plt.show()