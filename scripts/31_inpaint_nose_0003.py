import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    BASE_DIR
    / "output"
    / "single_test_0003"
    / "before.jpg"
)

MASK_PATH = (
    BASE_DIR
    / "output"
    / "single_test_0003"
    / "mask.png"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "single_test_0003"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("NOSE INPAINTING TEST - 0003")
print("=" * 70)

print("Device:", device)


if device == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK FILES
# ============================================================

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


if not MASK_PATH.exists():

    raise FileNotFoundError(
        f"Mask not found:\n{MASK_PATH}"
    )


# ============================================================
# LOAD IMAGE + MASK
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


mask = Image.open(
    MASK_PATH
).convert("L")


print(
    "Image size:",
    image.size
)

print(
    "Mask size:",
    mask.size
)


# ============================================================
# MATCH DIMENSIONS
# ============================================================

if image.size != mask.size:

    print(
        "Resizing mask to image size..."
    )

    mask = mask.resize(
        image.size,
        Image.Resampling.NEAREST
    )


# ============================================================
# MODEL
# ============================================================

MODEL_ID = (
    "runwayml/stable-diffusion-inpainting"
)


print()
print("-" * 70)
print("Loading inpainting model...")
print("The first run may download several GB.")
print("-" * 70)


# ============================================================
# LOAD PIPELINE
# ============================================================

if device == "cuda":

    pipe = (
        StableDiffusionInpaintPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    )

else:

    pipe = (
        StableDiffusionInpaintPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None
        )
    )


pipe = pipe.to(device)


# Reduce VRAM usage
pipe.enable_attention_slicing()


# ============================================================
# PROMPT
# ============================================================

prompt = (
    "a realistic portrait photo, "
    "natural human face, "
    "subtle refined nose shape, "
    "natural nose anatomy, "
    "natural skin texture, "
    "photorealistic, "
    "preserve identity, "
    "preserve facial proportions"
)


negative_prompt = (
    "deformed face, "
    "distorted face, "
    "extra nose, "
    "double nose, "
    "crooked nose, "
    "blurry, "
    "unrealistic skin, "
    "plastic skin, "
    "different person, "
    "changed identity, "
    "artifacts"
)


# ============================================================
# GENERATION
# ============================================================

print()
print("-" * 70)
print("Generating AFTER image...")
print("-" * 70)


with torch.inference_mode():

    result = pipe(

        prompt=prompt,

        negative_prompt=negative_prompt,

        image=image,

        mask_image=mask,

        num_inference_steps=25,

        guidance_scale=7.0,

        strength=0.55
    )


after = result.images[0]


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR
    / "after.jpg"
)


after.save(
    output_path,
    quality=95
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "Before:",
    IMAGE_PATH
)

print(
    "Mask:  ",
    MASK_PATH
)

print(
    "After: ",
    output_path
)

print("=" * 70)