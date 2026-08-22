from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_datasets"


print("=" * 60)
print("ANNOTATION VALIDATION")
print("=" * 60)


# ============================================================
# FFHQ
# ============================================================

ffhq = RAW / "FFHQ"

json_files = list(ffhq.rglob("*.json"))

print("\nFFHQ")
print("-" * 40)
print(f"JSON files: {len(json_files)}")

valid_json = 0
invalid_json = 0

for path in json_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)

        valid_json += 1

    except Exception as e:
        invalid_json += 1
        print(f"ERROR   {path.name} -> {e}")

print(f"Valid JSON files: {valid_json}")
print(f"Invalid JSON files: {invalid_json}")


# ============================================================
# FaceSynthetics
# ============================================================

face_synth = RAW / "FaceSynthetics"

txt_files = list(face_synth.rglob("*.txt"))

print("\nFaceSynthetics")
print("-" * 40)
print(f"TXT files: {len(txt_files)}")

valid_txt = 0
invalid_txt = 0

for path in txt_files:
    try:
        data = path.read_bytes()

        if len(data) > 0:
            valid_txt += 1
        else:
            invalid_txt += 1
            print(f"EMPTY   {path.name}")

    except Exception as e:
        invalid_txt += 1
        print(f"ERROR   {path.name} -> {e}")

print(f"Readable TXT files: {valid_txt}")
print(f"Invalid/empty TXT files: {invalid_txt}")


# ============================================================
# CelebAMask-HQ
# ============================================================

celeba = RAW / "CelebAMask-HQ"

png_files = list(celeba.rglob("*.png"))
txt_files = list(celeba.rglob("*.txt"))

print("\nCelebAMask-HQ")
print("-" * 40)
print(f"PNG files: {len(png_files)}")
print(f"TXT files: {len(txt_files)}")

# PNG files were already checked by the image integrity script.
print("PNG files were already validated as readable images.")

valid_celeba_txt = 0
invalid_celeba_txt = 0

for path in txt_files:
    try:
        data = path.read_bytes()

        if len(data) > 0:
            valid_celeba_txt += 1
        else:
            invalid_celeba_txt += 1
            print(f"EMPTY   {path.name}")

    except Exception as e:
        invalid_celeba_txt += 1
        print(f"ERROR   {path.name} -> {e}")

print(f"Readable TXT files: {valid_celeba_txt}")
print(f"Invalid/empty TXT files: {invalid_celeba_txt}")


# ============================================================
# SCUT-FBP5500
# ============================================================

scut = RAW / "SCUT-FBP5500"

pts_files = list(scut.rglob("*.pts"))
xlsx_files = list(scut.rglob("*.xlsx"))

print("\nSCUT-FBP5500")
print("-" * 40)
print(f"PTS files : {len(pts_files)}")
print(f"XLSX files: {len(xlsx_files)}")


# ------------------------------------------------------------
# PTS validation
# ------------------------------------------------------------
# IMPORTANT:
# .pts files may use an encoding that is not UTF-8.
# Therefore, we validate that the files exist and are
# non-empty by reading raw bytes instead of forcing UTF-8.
# ------------------------------------------------------------

valid_pts = 0
invalid_pts = 0

for path in pts_files:
    try:
        data = path.read_bytes()

        if len(data) > 0:
            valid_pts += 1
        else:
            invalid_pts += 1
            print(f"EMPTY   {path.name}")

    except Exception as e:
        invalid_pts += 1
        print(f"ERROR   {path.name} -> {e}")

print(f"Readable PTS files: {valid_pts}")
print(f"Invalid/empty PTS files: {invalid_pts}")


# ------------------------------------------------------------
# XLSX validation
# ------------------------------------------------------------

valid_xlsx = 0
invalid_xlsx = 0

for path in xlsx_files:
    try:
        data = path.read_bytes()

        if len(data) > 0:
            valid_xlsx += 1
        else:
            invalid_xlsx += 1
            print(f"EMPTY   {path.name}")

    except Exception as e:
        invalid_xlsx += 1
        print(f"ERROR   {path.name} -> {e}")

print(f"Readable XLSX files: {valid_xlsx}")
print(f"Invalid/empty XLSX files: {invalid_xlsx}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANNOTATION VALIDATION FINISHED")
print("=" * 60)

print("\nSummary:")
print(f"FFHQ JSON       : {valid_json}/{len(json_files)} valid")
print(f"FaceSynthetics  : {valid_txt}/{len(txt_files)} valid TXT")
print(f"CelebAMask-HQ   : {valid_celeba_txt}/{len(txt_files)} valid TXT")
print(f"SCUT PTS        : {valid_pts}/{len(pts_files)} valid")
print(f"SCUT XLSX       : {valid_xlsx}/{len(xlsx_files)} valid")

print("\nRaw datasets were not modified.")