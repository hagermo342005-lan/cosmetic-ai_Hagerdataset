import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "processed" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "resnet18_beauty_best.pth"
HISTORY_PATH = ROOT / "processed" / "deep_learning" / "training_history.csv"

OUTPUT_DIR = ROOT / "processed" / "deep_learning" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ResNet18 Prediction Visualization")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Dataset
# ============================================================

class BeautyDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = ROOT / row["Image_Path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        score = float(row["Mean_Beauty_Score"])

        return image, torch.tensor(score, dtype=torch.float32), row["Filename"]


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load test dataset
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_CSV)

print(f"Test records: {len(df)}")

dataset = BeautyDataset(df, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Load model
# ============================================================

print("\nLoading ResNet18...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 1)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# Predictions
# ============================================================

print("\nGenerating predictions...")

actual = []
predicted = []
filenames = []

with torch.no_grad():

    for images, scores, names in loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.squeeze(1).cpu().numpy()

        actual.extend(scores.numpy())
        predicted.extend(preds)
        filenames.extend(names)


actual = np.array(actual)
predicted = np.array(predicted)

errors = np.abs(actual - predicted)


# ============================================================
# Save predictions
# ============================================================

pred_df = pd.DataFrame({
    "Filename": filenames,
    "Actual_Score": actual,
    "Predicted_Score": predicted,
    "Absolute_Error": errors
})

pred_csv = OUTPUT_DIR / "resnet18_predictions.csv"

pred_df.to_csv(pred_csv, index=False)

print(f"\nPredictions saved to:")
print(pred_csv)


# ============================================================
# Actual vs Predicted
# ============================================================

print("\nCreating Actual vs Predicted plot...")

plt.figure(figsize=(8, 8))

plt.scatter(
    actual,
    predicted,
    alpha=0.5
)

plt.plot(
    [1, 5],
    [1, 5],
    linestyle="--"
)

plt.xlabel("Actual Beauty Score")
plt.ylabel("Predicted Beauty Score")
plt.title("ResNet18 - Actual vs Predicted")

plt.xlim(1, 5)
plt.ylim(1, 5)

plt.grid(True, alpha=0.3)

plt.tight_layout()

scatter_path = OUTPUT_DIR / "resnet18_actual_vs_predicted.png"

plt.savefig(scatter_path, dpi=200)
plt.close()

print(scatter_path)


# ============================================================
# Error distribution
# ============================================================

print("\nCreating error distribution...")

plt.figure(figsize=(8, 5))

plt.hist(
    errors,
    bins=30
)

plt.xlabel("Absolute Prediction Error")
plt.ylabel("Number of Images")
plt.title("ResNet18 Prediction Error Distribution")

plt.tight_layout()

error_path = OUTPUT_DIR / "resnet18_error_distribution.png"

plt.savefig(error_path, dpi=200)
plt.close()

print(error_path)


# ============================================================
# Training curves
# ============================================================

if HISTORY_PATH.exists():

    print("\nCreating training curves...")

    history = pd.read_csv(HISTORY_PATH)

    print("History columns:")
    print(history.columns.tolist())

    # Loss
    if "Train_Loss" in history.columns:

        plt.figure(figsize=(9, 5))

        plt.plot(
            history["Epoch"],
            history["Train_Loss"],
            label="Train Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        loss_path = OUTPUT_DIR / "training_loss.png"

        plt.savefig(loss_path, dpi=200)
        plt.close()

        print(loss_path)

    # Validation MAE
    if "Val_MAE" in history.columns:

        plt.figure(figsize=(9, 5))

        plt.plot(
            history["Epoch"],
            history["Val_MAE"],
            label="Validation MAE"
        )

        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Validation MAE")

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        mae_path = OUTPUT_DIR / "validation_mae.png"

        plt.savefig(mae_path, dpi=200)
        plt.close()

        print(mae_path)

    # Validation Pearson
    if "Val_Pearson" in history.columns:

        plt.figure(figsize=(9, 5))

        plt.plot(
            history["Epoch"],
            history["Val_Pearson"],
            label="Validation Pearson"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Pearson Correlation")
        plt.title("Validation Pearson Correlation")

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        pearson_path = OUTPUT_DIR / "validation_pearson.png"

        plt.savefig(pearson_path, dpi=200)
        plt.close()

        print(pearson_path)


# ============================================================
# Best / Worst predictions
# ============================================================

print("\nCreating prediction sample images...")

# Best predictions = smallest errors
best = pred_df.sort_values("Absolute_Error").head(12)

# Worst predictions = largest errors
worst = pred_df.sort_values("Absolute_Error", ascending=False).head(12)


def create_image_grid(dataframe, output_path, title):

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 11)
    )

    axes = axes.flatten()

    for ax, (_, row) in zip(axes, dataframe.iterrows()):

        image_path = ROOT / df.loc[
            df["Filename"] == row["Filename"],
            "Image_Path"
        ].iloc[0]

        image = Image.open(image_path).convert("RGB")

        ax.imshow(image)

        ax.set_title(
            f'{row["Filename"]}\n'
            f'Actual: {row["Actual_Score"]:.2f} | '
            f'Pred: {row["Predicted_Score"]:.2f}\n'
            f'Error: {row["Absolute_Error"]:.2f}'
        )

        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


best_path = OUTPUT_DIR / "best_predictions.png"

create_image_grid(
    best,
    best_path,
    "ResNet18 - Best Predictions"
)

print(best_path)


worst_path = OUTPUT_DIR / "worst_predictions.png"

create_image_grid(
    worst,
    worst_path,
    "ResNet18 - Worst Predictions"
)

print(worst_path)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Visualization Completed")
print("=" * 60)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*")):
    print(f"  {file.name}")

print("\n" + "=" * 60)