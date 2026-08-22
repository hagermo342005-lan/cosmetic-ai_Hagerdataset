import json
from pathlib import Path

import cv2
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "input" / "0003.png"

OUTPUT_DIR = BASE_DIR / "processed" / "landmarks_single"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = OUTPUT_DIR / "0003.json"


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


height, width = image.shape[:2]


print("=" * 70)
print("SINGLE IMAGE LANDMARK EXTRACTION")
print("=" * 70)

print(f"Image : {IMAGE_PATH}")
print(f"Size  : {width} x {height}")

print()


# ============================================================
# MEDIAPIPE
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"MediaPipe model not found:\n{MODEL_PATH}"
    )


BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,
)


# ============================================================
# CONVERT IMAGE
# ============================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)


# ============================================================
# DETECT LANDMARKS
# ============================================================

print("Detecting face landmarks...")

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    result = landmarker.detect(
        mp_image
    )


# ============================================================
# CHECK FACE
# ============================================================

if not result.face_landmarks:

    raise RuntimeError(
        "No face detected in the image."
    )


face_landmarks = (
    result.face_landmarks[0]
)


print(
    f"Landmarks detected: "
    f"{len(face_landmarks)}"
)


# ============================================================
# CONVERT TO PIXEL COORDINATES
# ============================================================

landmarks = []

for landmark in face_landmarks:

    landmarks.append(
        {
            "x": round(
                float(landmark.x * width),
                3
            ),

            "y": round(
                float(landmark.y * height),
                3
            ),

            "z": round(
                float(landmark.z),
                6
            ),
        }
    )


# ============================================================
# SAVE JSON
# ============================================================

data = {

    "image": IMAGE_PATH.name,

    "image_width": width,

    "image_height": height,

    "num_landmarks": len(
        landmarks
    ),

    "landmarks": landmarks,

    "source": "input/0003.png",

    "model": "MediaPipe Face Landmarker",

}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Image     : {IMAGE_PATH}"
)

print(
    f"Landmarks : {len(landmarks)}"
)

print(
    f"JSON      : {OUTPUT_PATH}"
)

print("=" * 70)