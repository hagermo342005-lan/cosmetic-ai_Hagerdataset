import cv2
import numpy as np

image_path = r".\processed\images\00000.png"
mask_path = r".\processed\masks\00000.png"

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

if mask is None:
    raise FileNotFoundError(f"Mask not found: {mask_path}")

if image.shape[:2] != mask.shape[:2]:
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

binary = mask > 0

overlay = image.copy()
overlay[binary] = (0, 255, 0)

result = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

cv2.imshow("Original", image)
cv2.imshow("Binary Mask", mask)
cv2.imshow("Mask Overlay", result)

cv2.waitKey(0)
cv2.destroyAllWindows()