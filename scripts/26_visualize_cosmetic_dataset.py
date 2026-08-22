import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "cosmetic_dataset",
    "train"
)

sample_ids = []

if os.path.exists(DATASET_DIR):

    for name in sorted(os.listdir(DATASET_DIR)):

        folder = os.path.join(DATASET_DIR, name)

        if not os.path.isdir(folder):
            continue

        before_path = os.path.join(folder, "before.jpg")
        after_path = os.path.join(folder, "after.jpg")
        mask_path = os.path.join(folder, "mask.png")

        if (
            os.path.isfile(before_path)
            and os.path.isfile(after_path)
            and os.path.isfile(mask_path)
        ):
            sample_ids.append(name)

sample_ids = sample_ids[:3]

print("=" * 70)
print("COSMETIC DATASET VISUALIZATION")
print("=" * 70)

print("Dataset:")
print(DATASET_DIR)

print()
print("Complete samples found:", len(sample_ids))

if len(sample_ids) == 0:

    print()
    print("No complete samples found.")
    print()
    print("Expected structure:")
    print("cosmetic_dataset/train/ID/before.jpg")
    print("cosmetic_dataset/train/ID/after.jpg")
    print("cosmetic_dataset/train/ID/mask.png")

    raise SystemExit

print()
print("Selected samples:")

for sid in sample_ids:
    print(sid)

print("=" * 70)

fig, axes = plt.subplots(
    len(sample_ids),
    3,
    figsize=(12, 4 * len(sample_ids))
)

if len(sample_ids) == 1:
    axes = [axes]

for row, sid in enumerate(sample_ids):

    folder = os.path.join(
        DATASET_DIR,
        sid
    )

    before_path = os.path.join(
        folder,
        "before.jpg"
    )

    after_path = os.path.join(
        folder,
        "after.jpg"
    )

    mask_path = os.path.join(
        folder,
        "mask.png"
    )

    before = plt.imread(before_path)
    mask = plt.imread(mask_path)
    after = plt.imread(after_path)

    axes[row][0].imshow(before)
    axes[row][0].set_title(sid + " - Before")
    axes[row][0].axis("off")

    axes[row][1].imshow(mask, cmap="gray")
    axes[row][1].set_title(sid + " - Mask")
    axes[row][1].axis("off")

    axes[row][2].imshow(after)
    axes[row][2].set_title(sid + " - After")
    axes[row][2].axis("off")

plt.tight_layout()
plt.show()