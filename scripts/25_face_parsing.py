from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES_DIR = ROOT / "processed" / "faces"
MASKS_DIR = ROOT / "processed" / "masks"

MASKS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FACE PARSING")
print("=" * 70)

print(f"Device : {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

print()
print(f"Model  : {MODEL_NAME}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading segmentation model...")

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("Model loaded.")
print()


# ============================================================
# MODEL LABELS
# ============================================================

id2label = model.config.id2label

print("Model classes:")
for idx, label in sorted(
    id2label.items(),
    key=lambda x: int(x[0])
):
    print(f"{idx:3} : {label}")

print()


# ============================================================
# INPUT IMAGES
# ============================================================

images = sorted(
    [
        p
        for p in FACES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)

print(f"Input faces : {len(images):,}")
print()


# ============================================================
# PROCESS
# ============================================================

success = 0
failed = 0

for image_path in tqdm(
    images,
    desc="Parsing faces"
):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        original_width, original_height = image.size

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(
                original_height,
                original_width
            ),
            mode="bilinear",
            align_corners=False
        )

        segmentation = logits.argmax(
            dim=1
        )[0]

        segmentation = (
            segmentation
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        stem = image_path.stem

        output_path = (
            MASKS_DIR /
            f"{stem}_parsing.png"
        )

        Image.fromarray(
            segmentation
        ).save(output_path)

        success += 1

    except Exception as e:

        failed += 1

        print(
            f"\nFAILED: {image_path.name}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": MODEL_NAME,
    "device": str(DEVICE),
    "input_images": len(images),
    "successful": success,
    "failed": failed,
    "classes": {
        str(k): v
        for k, v in id2label.items()
    }
}

with open(
    MASKS_DIR / "parsing_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("FACE PARSING COMPLETED")
print("=" * 70)

print(f"Input images : {len(images):,}")
print(f"Successful   : {success:,}")
print(f"Failed       : {failed:,}")

if images:

    print(
        f"Success rate : "
        f"{success / len(images) * 100:.2f}%"
    )

print()
print(
    f"Masks saved in:\n{MASKS_DIR}"
)

print(
    f"Metadata:\n"
    f"{MASKS_DIR / 'parsing_metadata.json'}"
)

print("=" * 70)