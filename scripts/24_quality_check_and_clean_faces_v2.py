import cv2
import json
import shutil
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FACES = ROOT / "processed" / "faces"
LANDMARKS = ROOT / "processed" / "landmarks_ffhq"
MASKS = ROOT / "processed" / "masks"

REJECTED_FACES = ROOT / "processed" / "faces_rejected"
REJECTED_LANDMARKS = ROOT / "processed" / "landmarks_ffhq_rejected"
REJECTED_MASKS = ROOT / "processed" / "masks_rejected"

REPORT = ROOT / "processed" / "faces_quality_report_v2.csv"

for p in [
    REJECTED_FACES,
    REJECTED_LANDMARKS,
    REJECTED_MASKS,
]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_landmarks(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pts = data.get("landmarks", [])

        if len(pts) < 80:
            return None

        arr = np.array(
            [[float(p["x"]), float(p["y"])] for p in pts],
            dtype=np.float32,
        )

        return arr

    except Exception:
        return None


def symmetry_score(gray):
    """
    Detect extremely artificial mirror/split-face images.

    Normal frontal faces can be somewhat symmetric, so we only
    reject very strong symmetry combined with a strong center seam.
    """

    h, w = gray.shape

    crop = gray[
        int(h * 0.08):int(h * 0.92),
        int(w * 0.10):int(w * 0.90)
    ]

    h2, w2 = crop.shape
    half = min(w2 // 2, 180)

    left = crop[:, :half]
    right = crop[:, -half:]

    right = cv2.flip(right, 1)

    if left.shape != right.shape:
        return 0.0, 0.0

    left_f = left.astype(np.float32)
    right_f = right.astype(np.float32)

    mae = np.mean(np.abs(left_f - right_f)) / 255.0

    # Convert to similarity
    similarity = 1.0 - mae

    # Center seam strength
    center = crop[:, w2 // 2 - 3:w2 // 2 + 3]
    left_center = crop[:, w2 // 2 - 12:w2 // 2 - 6]
    right_center = crop[:, w2 // 2 + 6:w2 // 2 + 12]

    seam = float(
        abs(
            center.mean()
            - (left_center.mean() + right_center.mean()) / 2
        )
    ) / 255.0

    return similarity, seam


def landmark_geometry_score(pts, w, h):
    """
    Basic geometry sanity check.
    Does not try to identify the person.
    """

    x = pts[:, 0]
    y = pts[:, 1]

    # Landmarks must be inside image
    inside = (
        (x >= 0) &
        (x < w) &
        (y >= 0) &
        (y < h)
    )

    if inside.mean() < 0.98:
        return False, "landmarks_out_of_bounds"

    # Bounding box
    bw = x.max() - x.min()
    bh = y.max() - y.min()

    if bw < w * 0.20:
        return False, "face_too_narrow"

    if bh < h * 0.25:
        return False, "face_too_short"

    if bw > w * 0.98:
        return False, "face_too_wide"

    if bh > h * 0.98:
        return False, "face_too_tall"

    # Face should not be extremely off-center
    cx = (x.min() + x.max()) / 2
    cy = (y.min() + y.max()) / 2

    if abs(cx - w / 2) > w * 0.32:
        return False, "face_off_center"

    if abs(cy - h / 2) > h * 0.35:
        return False, "face_off_center"

    return True, ""


def image_quality(gray):
    """
    Basic blur / contrast checks.
    """

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())

    # Very low sharpness
    if blur < 8:
        return False, "extreme_blur"

    # Almost no contrast
    if contrast < 10:
        return False, "very_low_contrast"

    return True, ""


# ============================================================
# PROCESS
# ============================================================

images = sorted(FACES.glob("*.jpg"))

print("=" * 70)
print("SECOND FFHQ FACE QUALITY CHECK")
print("=" * 70)
print(f"Input images : {len(images)}")
print(f"Faces        : {FACES}")
print("=" * 70)

valid = 0
rejected = 0

reasons_count = {}

rows = []

for i, img_path in enumerate(images, 1):

    image_id = img_path.stem

    reason = ""

    img = cv2.imread(str(img_path))

    if img is None:
        reason = "unreadable_image"

    else:
        h, w = img.shape[:2]

        # Must be 512x512
        if (w, h) != (512, 512):
            reason = "wrong_resolution"

        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ------------------------------------------------
            # 1. IMAGE QUALITY
            # ------------------------------------------------

            ok, r = image_quality(gray)

            if not ok:
                reason = r

            # ------------------------------------------------
            # 2. LANDMARKS
            # ------------------------------------------------

            if not reason:

                lm_path = LANDMARKS / f"{image_id}.json"

                if not lm_path.exists():
                    reason = "missing_landmarks"

                else:
                    pts = load_landmarks(lm_path)

                    if pts is None:
                        reason = "invalid_landmarks"

                    else:
                        ok, r = landmark_geometry_score(
                            pts, w, h
                        )

                        if not ok:
                            reason = r

            # ------------------------------------------------
            # 3. EXTREME MIRROR / SPLIT ARTIFACT
            # ------------------------------------------------

            if not reason:

                sym, seam = symmetry_score(gray)

                # Very strong left/right duplication plus
                # center seam = high-confidence mirror artifact.
                if sym > 0.965 and seam > 0.035:
                    reason = "mirror_split_artifact"

            # ------------------------------------------------
            # 4. MULTIPLE FACE HINT
            # ------------------------------------------------

            # Look for a second strong face-like region using
            # Haar cascade. This is only used as a rejection
            # signal when confidence is high.

            if not reason:

                cascade_path = cv2.data.haarcascades + \
                    "haarcascade_frontalface_default.xml"

                cascade = cv2.CascadeClassifier(cascade_path)

                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(70, 70)
                )

                if len(faces) > 1:
                    reason = "multiple_faces"

    # ========================================================
    # ACCEPT / REJECT
    # ========================================================

    if reason:

        rejected += 1
        reasons_count[reason] = reasons_count.get(reason, 0) + 1

        # Move face
        destination = REJECTED_FACES / img_path.name
        shutil.move(str(img_path), str(destination))

        # Move landmarks if present
        lm = LANDMARKS / f"{image_id}.json"

        if lm.exists():
            shutil.move(
                str(lm),
                str(REJECTED_LANDMARKS / lm.name)
            )

        # Move mask if present
        mask = MASKS / f"{image_id}_mask.png"

        if mask.exists():
            shutil.move(
                str(mask),
                str(REJECTED_MASKS / mask.name)
            )

        status = "REJECTED"

    else:
        valid += 1
        status = "VALID"

    rows.append(
        f"{image_id},{status},{reason}"
    )

    if i % 100 == 0:
        print(
            f"Processed: {i} | "
            f"Valid: {valid} | "
            f"Rejected: {rejected}"
        )


# ============================================================
# REPORT
# ============================================================

with open(REPORT, "w", encoding="utf-8") as f:

    f.write("id,status,reason\n")

    for row in rows:
        f.write(row + "\n")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("SECOND QUALITY CHECK COMPLETED")
print("=" * 70)

print(f"Checked images : {len(images)}")
print(f"Valid images   : {valid}")
print(f"Rejected       : {rejected}")

if len(images):
    print(
        f"Valid rate     : "
        f"{valid / len(images) * 100:.2f}%"
    )

print()
print("Rejection reasons:")

for reason, count in sorted(
    reasons_count.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"  {reason:25s}: {count}")

print()
print("Valid faces:")
print(FACES)

print()
print("Rejected faces:")
print(REJECTED_FACES)

print()
print("Report:")
print(REPORT)

print("=" * 70)