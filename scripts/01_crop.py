import cv2
import os

INPUT = "raw_datasets/FFHQ/images/00000.png"
OUTPUT = "processed/faces/00000.png"

os.makedirs("processed/faces", exist_ok=True)

img = cv2.imread(INPUT)

if img is None:
    raise FileNotFoundError(
        f"Image not found: {INPUT}"
    )

face = cv2.resize(img, (512, 512))

cv2.imwrite(OUTPUT, face)

print("Saved:", OUTPUT)