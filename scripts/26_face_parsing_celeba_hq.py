from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = (
    ROOT
    / "processed"
    / "faces"
    / "CelebAMask-HQ"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "face_parsing"
    / "parsing_resnet18.onnx"
)

RAW_MASKS_DIR = (
    ROOT
    / "processed"
    / "face_parsing_masks"
)

COLOR_DIR = (
    ROOT
    / "processed"
    / "face_parsing_colored"
)


RAW_MASKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COLOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 512
NUM_CLASSES = 19

# Visualization colors
COLORS = np.array([
    [0,   0,   0],
    [255, 0,   0],
    [0,   255, 0],
    [0,   0,   255],
    [255, 255, 0],
    [255, 0,   255],
    [0,   255, 255],
    [128, 0,   0],
    [0,   128, 0],
    [0,   0,   128],
    [128, 128, 0],
    [128, 0,   128],
    [0,   128, 128],
    [255, 128, 0],
    [255, 0,   128],
    [128, 255, 0],
    [0,   255, 128],
    [128, 128, 255],
    [255, 128, 128],
], dtype=np.uint8)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CELEBAMASK-HQ FACE PARSING")
print("=" * 70)

print(f"Faces       : {FACES_DIR}")
print(f"Model       : {MODEL_PATH}")
print(f"Raw masks   : {RAW_MASKS_DIR}")
print(f"Colored     : {COLOR_DIR}")
print()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ONNX model...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name

print("Model loaded.")

print(
    f"Input name : {input_name}"
)

print()


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    [
        p
        for p in FACES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]
)


print(
    f"Images found : {len(images):,}"
)

print("=" * 70)


# ============================================================
# COUNTERS
# ============================================================

success = 0
failed = 0
skipped = 0


# ============================================================
# PROCESS
# ============================================================

for image_path in tqdm(
    images,
    desc="Face Parsing"
):

    stem = image_path.stem

    raw_path = (
        RAW_MASKS_DIR
        / f"{stem}.png"
    )

    color_path = (
        COLOR_DIR
        / f"{stem}_colored.jpg"
    )


    # --------------------------------------------------------
    # Skip if already processed
    # --------------------------------------------------------

    if (
        raw_path.exists()
        and
        color_path.exists()
    ):

        skipped += 1
        continue


    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            failed += 1

            continue


        original_height, original_width = (
            image.shape[:2]
        )


        # ====================================================
        # BGR -> RGB
        # ====================================================

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # RESIZE
        # ====================================================

        resized = cv2.resize(
            image_rgb,
            (
                INPUT_SIZE,
                INPUT_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )


        # ====================================================
        # NORMALIZE
        # ====================================================

        input_image = (
            resized.astype(
                np.float32
            )
            / 255.0
        )


        # HWC -> CHW

        input_image = np.transpose(
            input_image,
            (2, 0, 1)
        )


        # Add batch dimension

        input_image = input_image[
            None,
            ...
        ]


        # ====================================================
        # INFERENCE
        # ====================================================

        outputs = session.run(
            None,
            {
                input_name:
                input_image
            }
        )


        # ====================================================
        # FIRST OUTPUT
        # ====================================================

        logits = outputs[0]


        # Expected:
        # (1, 19, 512, 512)

        if logits.ndim != 4:

            raise RuntimeError(
                f"Unexpected output shape: "
                f"{logits.shape}"
            )


        if logits.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                f"Expected {NUM_CLASSES} classes "
                f"but got {logits.shape[1]}"
            )


        # ====================================================
        # ARGMAX
        # ====================================================

        segmentation = np.argmax(
            logits[0],
            axis=0
        ).astype(
            np.uint8
        )


        # ====================================================
        # RESIZE MASK TO ORIGINAL SIZE
        # ====================================================

        segmentation = cv2.resize(
            segmentation,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_NEAREST
        )


        # ====================================================
        # SAVE RAW MASK
        # ====================================================

        cv2.imwrite(
            str(raw_path),
            segmentation
        )


        # ====================================================
        # CREATE COLORED MASK
        # ====================================================

        colored_mask = COLORS[
            segmentation
        ]


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = cv2.addWeighted(
            image,
            0.55,
            colored_mask,
            0.45,
            0
        )


        # ====================================================
        # SAVE COLORED VISUALIZATION
        # ====================================================

        cv2.imwrite(
            str(color_path),
            overlay
        )


        success += 1


    except Exception as e:

        failed += 1

        print(
            f"\nFAILED: "
            f"{image_path.name}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

total = len(images)

processed_total = (
    success
    +
    skipped
)

success_rate = (
    processed_total
    /
    total
    *
    100
    if total
    else 0
)


print()

print("=" * 70)
print(
    "FACE PARSING COMPLETED"
)
print("=" * 70)

print(
    f"Input images     : {total:,}"
)

print(
    f"Newly processed  : {success:,}"
)

print(
    f"Already existed  : {skipped:,}"
)

print(
    f"Failed           : {failed:,}"
)

print(
    f"Available result : "
    f"{processed_total:,}"
)

print(
    f"Coverage         : "
    f"{success_rate:.2f}%"
)

print()

print(
    "Raw masks:"
)

print(
    RAW_MASKS_DIR
)

print()

print(
    "Colored visualizations:"
)

print(
    COLOR_DIR
)

print()

print("=" * 70)