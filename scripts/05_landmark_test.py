import cv2
import json
from pathlib import Path

FACE_DIR = Path("processed/faces")
LANDMARK_DIR = Path("processed/landmarks")
OUTPUT_DIR = Path("processed/landmark_test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_path = FACE_DIR / "00000.png"
json_path = LANDMARK_DIR / "00000.json"

image = cv2.imread(str(image_path))

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

landmarks = data["landmarks"]

h, w = image.shape[:2]

# رسم كل landmarks
for lm in landmarks:

    x = int(lm["x"] * w)
    y = int(lm["y"] * h)

    if 0 <= x < w and 0 <= y < h:
        cv2.circle(
            image,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

output = OUTPUT_DIR / "00000_landmarks.jpg"

cv2.imwrite(str(output), image)

print(f"Saved: {output}")