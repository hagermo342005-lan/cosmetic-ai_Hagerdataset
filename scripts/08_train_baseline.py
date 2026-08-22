from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import joblib

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ROOT / "processed" / "splits" / "train.csv"
VAL = ROOT / "processed" / "splits" / "validation.csv"
TEST = ROOT / "processed" / "splits" / "test.csv"

LANDMARK_DIR = ROOT / "processed" / "landmarks"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features(df):
    X = []
    y = []

    for _, row in df.iterrows():
        landmark_file = LANDMARK_DIR / (
            Path(row["Filename"]).stem + ".json"
        )

        with open(landmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data["landmarks"]

        if len(points) != 86:
            continue

        # Normalize landmarks by face bounding box
        points = np.array(
            [[p["x"], p["y"]] for p in points],
            dtype=np.float32
        )

        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)

        center = (min_xy + max_xy) / 2
        scale = max(max_xy - min_xy)

        if scale == 0:
            continue

        points = (points - center) / scale

        X.append(points.flatten())
        y.append(row["Mean_Beauty_Score"])

    return np.array(X), np.array(y)


print("=" * 60)
print("Training Baseline Beauty Model")
print("=" * 60)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN)
val_df = pd.read_csv(VAL)
test_df = pd.read_csv(TEST)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

print("\nExtracting landmark features...")

X_train, y_train = load_features(train_df)
X_val, y_val = load_features(val_df)
X_test, y_test = load_features(test_df)

print(f"\nTraining features : {X_train.shape}")
print(f"Validation        : {X_val.shape}")
print(f"Test              : {X_test.shape}")

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


def evaluate(name, X, y):
    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    correlation = pearsonr(y, predictions)[0]

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE                 : {mae:.4f}")
    print(f"RMSE                : {rmse:.4f}")
    print(f"Pearson Correlation : {correlation:.4f}")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Pearson": float(correlation),
    }


results = {
    "validation": evaluate("Validation", X_val, y_val),
    "test": evaluate("Test", X_test, y_test),
}

model_path = OUTPUT_DIR / "baseline_landmark_model.joblib"

joblib.dump(model, model_path)

results_path = OUTPUT_DIR / "baseline_results.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Baseline Training Completed")
print("=" * 60)

print(f"Model saved to:")
print(model_path)

print(f"\nResults saved to:")
print(results_path)

print("=" * 60)