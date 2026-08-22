import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

INPUT = Path("input/00002.png")
OUTPUT = Path("output/single_test")

OUTPUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load image
# -----------------------------
image = cv2.imread(str(INPUT))

if image is None:
    raise FileNotFoundError(f"Image not found: {INPUT}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]

# -----------------------------
# MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    result = face_mesh.process(rgb)

if not result.multi_face_landmarks:
    raise RuntimeError("No face detected.")

face = result.multi_face_landmarks[0]

# -----------------------------
# Save BEFORE
# -----------------------------
cv2.imwrite(
    str(OUTPUT / "before.jpg"),
    image
)

# -----------------------------
# Save landmarks
# -----------------------------
landmarks = []

for i, p in enumerate(face.landmark):

    x = int(p.x * w)
    y = int(p.y * h)

    landmarks.append({
        "id": i,
        "x": x,
        "y": y,
        "z": float(p.z)
    })

with open(
    OUTPUT / "landmarks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        landmarks,
        f,
        indent=2
    )

# -----------------------------
# Draw landmarks
# -----------------------------
visual = image.copy()

for p in landmarks:

    cv2.circle(
        visual,
        (p["x"], p["y"]),
        1,
        (0, 255, 0),
        -1
    )

cv2.imwrite(
    str(OUTPUT / "landmarks.jpg"),
    visual
)

# -----------------------------
# Nose region
# -----------------------------
nose_indices = [
    1, 2, 4, 5,
    6, 19, 94,
    97, 98, 168,
    195, 197,
    326, 327, 344,
    440
]

points = np.array(
    [
        [landmarks[i]["x"], landmarks[i]["y"]]
        for i in nose_indices
    ],
    dtype=np.int32
)

# -----------------------------
# Nose mask
# -----------------------------
mask = np.zeros(
    (h, w),
    dtype=np.uint8
)

hull = cv2.convexHull(points)

cv2.fillConvexPoly(
    mask,
    hull,
    255
)

# Expand mask slightly
kernel = np.ones((9, 9), np.uint8)

mask = cv2.dilate(
    mask,
    kernel,
    iterations=2
)

cv2.imwrite(
    str(OUTPUT / "mask.png"),
    mask
)

# -----------------------------
# Overlay mask
# -----------------------------
overlay = image.copy()

red = np.zeros_like(image)
red[:, :, 2] = 255

mask_bool = mask > 0

overlay[mask_bool] = cv2.addWeighted(
    image[mask_bool],
    0.5,
    red[mask_bool],
    0.5,
    0
)

cv2.imwrite(
    str(OUTPUT / "mask_overlay.jpg"),
    overlay
)

print("=" * 60)
print("SINGLE FACE TEST COMPLETED")
print("=" * 60)
print(f"Input:          {INPUT}")
print(f"Before:         {OUTPUT / 'before.jpg'}")
print(f"Landmarks:      {OUTPUT / 'landmarks.jpg'}")
print(f"Mask:            {OUTPUT / 'mask.png'}")
print(f"Mask overlay:   {OUTPUT / 'mask_overlay.jpg'}")
print(f"Landmarks JSON: {OUTPUT / 'landmarks.json'}")
print("=" * 60)