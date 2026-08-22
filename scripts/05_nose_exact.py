import cv2
import json
import numpy as np
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

IMAGE_PATH = Path("processed/faces/00000.png")
LANDMARK_PATH = Path("processed/landmarks/00000.json")

OUT_DIR = Path("processed/nose_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD IMAGE
# ==========================================

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise FileNotFoundError(IMAGE_PATH)

h, w = img.shape[:2]

# ==========================================
# LOAD LANDMARKS
# ==========================================

with open(LANDMARK_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

landmarks = data["landmarks"]

pts = np.array(
    [
        [
            int(p["x"] * w),
            int(p["y"] * h)
        ]
        for p in landmarks
    ],
    dtype=np.int32
)

print("Image:", IMAGE_PATH)
print("Landmarks:", len(pts))

# ==========================================================
# NOSE LANDMARKS
#
# These are arranged as connected anatomical regions.
# We DO NOT use convexHull.
# ==========================================================

nose_groups = {

    # Left side of nose
    "left_side": [
        98,
        97,
        326,
        327
    ],

    # Nose bridge
    "bridge": [
        168,
        6,
        197,
        195,
        5
    ],

    # Left nose contour
    "left_contour": [
        98,
        97,
        2,
        94,
        141,
        125
    ],

    # Nose tip
    "tip": [
        1,
        2,
        4
    ],

    # Right nose contour
    "right_contour": [
        4,
        275,
        327,
        326,
        355,
        392
    ],

    # Bottom / nostril area
    "bottom": [
        141,
        94,
        1,
        2,
        4,
        275,
        392
    ]
}

# ==========================================================
# DRAW ONLY LINES
# ==========================================================

result = img.copy()

# line thickness
THICKNESS = 2

# ==========================================
# Helper function
# ==========================================

def draw_connected(points_ids, color=(0, 255, 0)):
    valid = [i for i in points_ids if 0 <= i < len(pts)]

    for a, b in zip(valid[:-1], valid[1:]):

        x1, y1 = pts[a]
        x2, y2 = pts[b]

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            color,
            THICKNESS,
            cv2.LINE_AA
        )

# ==========================================================
# DRAW ANATOMICAL LINES
# ==========================================================

# Nose bridge
draw_connected(
    nose_groups["bridge"]
)

# Left contour
draw_connected(
    nose_groups["left_contour"]
)

# Right contour
draw_connected(
    nose_groups["right_contour"]
)

# Bottom of nose
draw_connected(
    nose_groups["bottom"]
)

# ==========================================================
# CONNECT IMPORTANT POINTS
# ==========================================================

connections = [

    # bridge → tip
    (6, 4),

    # left → tip
    (98, 1),

    # right → tip
    (327, 1),

    # bottom left → bottom center
    (141, 1),

    # bottom center → bottom right
    (1, 392),

    # nostril connections
    (125, 141),
    (355, 392),
]

for a, b in connections:

    if a < len(pts) and b < len(pts):

        x1, y1 = pts[a]
        x2, y2 = pts[b]

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            THICKNESS,
            cv2.LINE_AA
        )

# ==========================================================
# DRAW LANDMARK POINTS
# ==========================================================

all_ids = set()

for group in nose_groups.values():
    all_ids.update(group)

for idx in all_ids:

    if idx >= len(pts):
        continue

    x, y = pts[idx]

    cv2.circle(
        result,
        (x, y),
        2,
        (0, 0, 255),
        -1
    )

# ==========================================================
# SAVE
# ==========================================================

output = OUT_DIR / "00000_nose_outline.png"

cv2.imwrite(
    str(output),
    result
)

print()
print("==============================")
print("NOSE OUTLINE COMPLETE")
print("==============================")
print("Saved:")
print(output)
print("==============================")