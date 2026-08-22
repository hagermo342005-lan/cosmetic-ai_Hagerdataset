import cv2
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

# =========================
# FILES
# =========================

IMAGE = Path("processed/faces/00000.png")
LANDMARKS = Path("processed/landmarks/00000.json")
MODEL = Path("models/face_parsing/parsing_resnet18.onnx")

OUT = Path("processed/nose_test")
OUT.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD IMAGE
# =========================

img = cv2.imread(str(IMAGE))

if img is None:
    raise RuntimeError(f"Cannot read image: {IMAGE}")

h, w = img.shape[:2]

print("Image:", IMAGE)
print("Size:", w, "x", h)


# =========================
# LOAD LANDMARKS
# =========================

with open(LANDMARKS, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON type:", type(data))

# Find landmarks
if isinstance(data, dict):

    if "landmarks" in data:
        points = data["landmarks"]

    elif "points" in data:
        points = data["points"]

    elif "face_landmarks" in data:
        points = data["face_landmarks"]

    else:
        # print keys so we know exact structure
        print("JSON keys:", data.keys())

        raise RuntimeError(
            "Could not find landmarks in JSON"
        )

else:
    points = data


print("Number of landmarks:", len(points))


# =========================
# DRAW ALL LANDMARKS
# =========================

landmark_img = img.copy()

for p in points:

    if isinstance(p, dict):

        x = p.get("x")
        y = p.get("y")

    else:

        x = p[0]
        y = p[1]

    if x is None or y is None:
        continue

    # normalized coordinates
    if 0 <= x <= 1 and 0 <= y <= 1:

        px = int(x * w)
        py = int(y * h)

    else:

        px = int(x)
        py = int(y)

    if 0 <= px < w and 0 <= py < h:

        cv2.circle(
            landmark_img,
            (px, py),
            2,
            (0, 255, 0),
            -1
        )


cv2.imwrite(
    str(OUT / "01_all_landmarks.jpg"),
    landmark_img
)


# =========================
# FACE PARSING
# =========================

session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model input:", input_name)
print("Model shape:", session.get_inputs()[0].shape)


rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

resized = cv2.resize(
    rgb,
    (512, 512)
)

x = resized.astype(np.float32) / 255.0

x = np.transpose(
    x,
    (2, 0, 1)
)

x = np.expand_dims(
    x,
    0
)

output = session.run(
    None,
    {input_name: x}
)[0]

print("Output shape:", output.shape)

output = np.squeeze(output)

if output.ndim == 3:

    parsing = np.argmax(
        output,
        axis=0
    )

elif output.ndim == 2:

    parsing = output

else:

    raise RuntimeError(
        f"Unexpected output: {output.shape}"
    )


parsing = cv2.resize(
    parsing.astype(np.uint8),
    (w, h),
    interpolation=cv2.INTER_NEAREST
)


# =========================
# SAVE RAW PARSING
# =========================

cv2.imwrite(
    str(OUT / "02_parsing.png"),
    parsing
)


# =========================
# SHOW CLASS DISTRIBUTION
# =========================

classes, counts = np.unique(
    parsing,
    return_counts=True
)

print()
print("Classes found:")

for c, n in zip(classes, counts):

    print(
        f"class {c}: {n} pixels"
    )


# =========================
# TEST NOSE CLASS
# =========================

NOSE_CLASS = 2

nose = np.zeros(
    (h, w),
    dtype=np.uint8
)

nose[parsing == NOSE_CLASS] = 255


cv2.imwrite(
    str(OUT / "03_nose_class.png"),
    nose
)


# =========================
# OVERLAY
# =========================

overlay = img.copy()

red = np.zeros_like(img)

red[:, :, 2] = 255

mask = nose > 0

overlay[mask] = cv2.addWeighted(
    img[mask],
    0.35,
    red[mask],
    0.65,
    0
)


cv2.imwrite(
    str(OUT / "04_nose_overlay.jpg"),
    overlay
)


print()
print("==============================")
print("TEST COMPLETE")
print("==============================")
print("Results:")
print(OUT)
print("==============================")