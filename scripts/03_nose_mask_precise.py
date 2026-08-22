import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json

# =========================
# PATHS
# =========================

FACES_DIR = Path("processed/faces")
LANDMARKS_DIR = Path("processed/landmarks")
MODEL_PATH = Path("models/face_parsing/parsing_resnet18.onnx")

MASK_DIR = Path("processed/nose_masks_precise")
PREVIEW_DIR = Path("processed/nose_masks_precise_preview")

MASK_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# FACE PARSING
# =========================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]

print("Model input:", input_info.name)
print("Model shape:", input_info.shape)


# =========================
# CELEBAMASK-HQ LABEL
# =========================
# CelebAMask-HQ:
# 0 = background
# 1 = skin
# 2 = nose
#
# IMPORTANT:
# nose = class 2

NOSE_CLASS = 2


def run_parsing(image):

    original_h, original_w = image.shape[:2]

    # Model expects RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(rgb, (512, 512))

    # Normalize
    x = resized.astype(np.float32) / 255.0

    # CHW
    x = np.transpose(x, (2, 0, 1))

    # Batch
    x = np.expand_dims(x, axis=0)

    # inference
    output = session.run(
        None,
        {input_info.name: x}
    )[0]

    # Handle output
    output = np.squeeze(output)

    # If output is [C,H,W]
    if output.ndim == 3:
        segmentation = np.argmax(output, axis=0)

    # If output is already [H,W]
    elif output.ndim == 2:
        segmentation = output

    else:
        raise RuntimeError(
            f"Unexpected model output shape: {output.shape}"
        )

    # Resize segmentation to original
    segmentation = cv2.resize(
        segmentation.astype(np.uint8),
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    return segmentation


# =========================
# LANDMARK FILTER
# =========================

def get_nose_landmarks(data, width, height):

    points = data

    # MediaPipe Face Landmarker indices
    #
    # These points describe the nose region:
    #
    # nose bridge
    # nose sides
    # nose tip
    # nostril area

    indices = [
        1,      # nose tip / center
        2,
        4,
        5,
        6,
        19,
        20,
        44,
        45,
        48,
        49,
        51,
        94,
        97,
        98,
        115,
        122,
        129,
        168,
        195,
        197,
        236,
        240,
        278,
        290,
        326,
        327,
        344,
        351,
        419,
        420,
        429,
        437,
        440,
        456
    ]

    pts = []

    for idx in indices:

        if idx >= len(points):
            continue

        p = points[idx]

        # normalized coordinates
        x = int(p["x"] * width)
        y = int(p["y"] * height)

        if 0 <= x < width and 0 <= y < height:
            pts.append([x, y])

    return np.array(pts, dtype=np.int32)


# =========================
# CREATE PRECISE NOSE MASK
# =========================

def create_precise_mask(image, segmentation, landmark_points):

    h, w = image.shape[:2]

    # ---------------------------------
    # 1. Parsing nose
    # ---------------------------------

    parsing_mask = np.zeros((h, w), dtype=np.uint8)

    parsing_mask[segmentation == NOSE_CLASS] = 255

    # ---------------------------------
    # 2. Landmark polygon
    # ---------------------------------

    landmark_mask = np.zeros((h, w), dtype=np.uint8)

    if len(landmark_points) >= 3:

        hull = cv2.convexHull(landmark_points)

        cv2.fillConvexPoly(
            landmark_mask,
            hull,
            255
        )

    # ---------------------------------
    # 3. Intersection
    # ---------------------------------

    precise_mask = cv2.bitwise_and(
        parsing_mask,
        landmark_mask
    )

    # ---------------------------------
    # 4. Clean noise
    # ---------------------------------

    kernel = np.ones((3, 3), np.uint8)

    precise_mask = cv2.morphologyEx(
        precise_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    precise_mask = cv2.morphologyEx(
        precise_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ---------------------------------
    # 5. Keep only meaningful components
    # ---------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        precise_mask,
        connectivity=8
    )

    clean = np.zeros_like(precise_mask)

    if num_labels > 1:

        areas = stats[1:, cv2.CC_STAT_AREA]

        # largest component
        largest_idx = np.argmax(areas) + 1

        if areas[largest_idx - 1] > 100:

            clean[labels == largest_idx] = 255

    return clean, parsing_mask, landmark_mask


# =========================
# PREVIEW
# =========================

def create_preview(image, mask, landmark_mask):

    overlay = image.copy()

    # Red mask
    red = np.zeros_like(image)
    red[:, :, 2] = 255

    mask_bool = mask > 0

    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool],
        0.35,
        red[mask_bool],
        0.65,
        0
    )

    # Draw contour
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    return overlay


# =========================
# PROCESS
# =========================

files = sorted(FACES_DIR.glob("*.png"))

print()
print("==============================")
print("PRECISE NOSE MASK")
print("==============================")
print("Images:", len(files))
print()

success = 0
failed = 0

for i, image_path in enumerate(files):

    try:

        image = cv2.imread(str(image_path))

        if image is None:
            failed += 1
            continue

        h, w = image.shape[:2]

        image_id = image_path.stem

        # -------------------------
        # landmarks
        # -------------------------

        landmark_file = LANDMARKS_DIR / f"{image_id}.json"

        if not landmark_file.exists():

            print(
                f"[SKIP] {image_id}: landmark missing"
            )

            failed += 1
            continue

        with open(
            landmark_file,
            "r",
            encoding="utf-8"
        ) as f:

            landmark_data = json.load(f)

        # Support different JSON formats

        if isinstance(landmark_data, dict):

            if "landmarks" in landmark_data:
                points = landmark_data["landmarks"]

            elif "points" in landmark_data:
                points = landmark_data["points"]

            else:
                points = landmark_data

        else:
            points = landmark_data

        landmark_points = get_nose_landmarks(
            points,
            w,
            h
        )

        if len(landmark_points) < 3:

            print(
                f"[SKIP] {image_id}: insufficient landmarks"
            )

            failed += 1
            continue

        # -------------------------
        # parsing
        # -------------------------

        segmentation = run_parsing(image)

        # -------------------------
        # precise mask
        # -------------------------

        mask, parsing_mask, landmark_mask = create_precise_mask(
            image,
            segmentation,
            landmark_points
        )

        # -------------------------
        # check mask
        # -------------------------

        area = np.count_nonzero(mask)

        if area < 100:

            print(
                f"[SKIP] {image_id}: nose mask too small"
            )

            failed += 1
            continue

        # -------------------------
        # save mask
        # -------------------------

        mask_path = MASK_DIR / f"{image_id}.png"

        cv2.imwrite(
            str(mask_path),
            mask
        )

        # -------------------------
        # preview
        # -------------------------

        preview = create_preview(
            image,
            mask,
            landmark_mask
        )

        preview_path = PREVIEW_DIR / f"{image_id}.jpg"

        cv2.imwrite(
            str(preview_path),
            preview
        )

        success += 1

        if success % 100 == 0:

            print(
                f"Processed: {success}"
            )

    except Exception as e:

        failed += 1

        print(
            f"[ERROR] {image_path.name}: {e}"
        )


print()
print("==============================")
print("NOSE MASK COMPLETE")
print("==============================")
print("Success :", success)
print("Failed  :", failed)
print("Total   :", len(files))
print()
print("Masks:")
print(MASK_DIR)
print()
print("Preview:")
print(PREVIEW_DIR)
print("==============================")