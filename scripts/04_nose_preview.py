import cv2
from pathlib import Path

FACE_DIR = Path("processed/faces")
MASK_DIR = Path("processed/masks")
OUTPUT_DIR = Path("processed/nose_preview")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# أول 20 mask موجودين فعلًا
mask_files = list(MASK_DIR.glob("*.png"))[:20]

for mask_path in mask_files:

    image_id = mask_path.stem

    face_path = FACE_DIR / f"{image_id}.png"

    if not face_path.exists():
        continue

    image = cv2.imread(str(face_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    # Overlay
    overlay = image.copy()

    red = image.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0
    red[:, :, 2] = 255

    mask_area = mask > 0

    overlay[mask_area] = cv2.addWeighted(
        image[mask_area],
        0.35,
        red[mask_area],
        0.65,
        0
    )

    # Save
    output = OUTPUT_DIR / f"{image_id}_nose_preview.jpg"

    cv2.imwrite(str(output), overlay)

    print(f"Saved: {output}")

print("DONE")