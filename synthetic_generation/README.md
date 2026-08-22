# Synthetic Generation

This directory contains the pipeline and configuration used to
generate synthetic cosmetic surgery before/after images.

The generation pipeline uses:

- Processed face images
- Facial landmarks
- Target-region masks
- Operation conditions
- Image-to-image or inpainting generation
- SDXL / ControlNet / LoRA when applicable

Generated images should include metadata describing:

- ID
- Operation
- Source type
- Generator
- Seed
- Input files
- Quality flag

Large generated images and model checkpoints are excluded from GitHub.
