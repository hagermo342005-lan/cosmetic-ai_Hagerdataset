import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = Path("input/00002.png")
MASK_PATH = Path("output/single_test/mask.png")
OUTPUT_DIR = Path("output/single_test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("NOSE INPAINTING TEST")
print("=" * 65)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# LOAD IMAGE + MASK
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")
mask = Image.open(MASK_PATH).convert("L")

print("Image size:", image.size)
print("Mask size:", mask.size)

# Make sure dimensions match
if image.size != mask.size:
    mask = mask.resize(
        image.size,
        Image.Resampling.NEAREST
    )

# ============================================================
# LOAD INPAINTING MODEL
# ============================================================

MODEL_ID = "runwayml/stable-diffusion-inpainting"

print("-" * 65)
print("Loading model...")
print("The first run may download several GB.")
print("-" * 65)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to(device)

# Reduce VRAM usage
pipe.enable_attention_slicing()

# ============================================================
# PROMPT
# ============================================================

prompt = (
    "a realistic portrait photo, natural human face, "
    "subtle refined nose shape, natural skin texture, "
    "photorealistic, preserve identity"
)

negative_prompt = (
    "deformed face, distorted face, extra nose, "
    "blurry, unrealistic skin, plastic skin, "
    "different person, artifacts"
)

# ============================================================
# GENERATE
# ============================================================

print("-" * 65)
print("Generating AFTER image...")
print("-" * 65)

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

output_path = OUTPUT_DIR / "after.jpg"

after.save(output_path, quality=95)

print("=" * 65)
print("SUCCESS")
print("=" * 65)
print("Before:", IMAGE_PATH)
print("Mask:  ", MASK_PATH)
print("After: ", output_path)
print("=" * 65)