from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "processed" / "splits" / "train.csv"
VAL_CSV = ROOT / "processed" / "splits" / "validation.csv"
TEST_CSV = ROOT / "processed" / "splits" / "test.csv"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "processed" / "deep_learning"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_beauty_best.pth"
RESULTS_PATH = OUTPUT_DIR / "resnet18_results.json"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ResNet18 Beauty Prediction")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ============================================================
# Dataset
# ============================================================

class BeautyDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = ROOT / row["Image_Path"]

        image = Image.open(image_path).convert("RGB")

        score = float(row["Mean_Beauty_Score"])

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(score, dtype=torch.float32)

        return image, target


# ============================================================
# Load CSV files
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")
print()


# ============================================================
# Verify image paths
# ============================================================

def filter_existing_images(df, name):

    valid = []

    for _, row in df.iterrows():

        path = ROOT / row["Image_Path"]

        if path.exists():
            valid.append(True)
        else:
            valid.append(False)

    df = df.loc[valid].reset_index(drop=True)

    print(f"{name} valid images: {len(df)}")

    return df


train_df = filter_existing_images(train_df, "Train")
val_df = filter_existing_images(val_df, "Validation")
test_df = filter_existing_images(test_df, "Test")

print()


# ============================================================
# Image transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DataLoaders
# ============================================================

train_dataset = BeautyDataset(
    train_df,
    transform=train_transform
)

val_dataset = BeautyDataset(
    val_df,
    transform=eval_transform
)

test_dataset = BeautyDataset(
    test_df,
    transform=eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Model
# ============================================================

print("Loading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 1)

model = model.to(DEVICE)

print("Model ready.")
print()


# ============================================================
# Loss / Optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(predictions, targets):

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    mae = np.mean(np.abs(predictions - targets))

    rmse = np.sqrt(
        np.mean((predictions - targets) ** 2)
    )

    if np.std(predictions) == 0 or np.std(targets) == 0:
        pearson = 0.0
    else:
        pearson = np.corrcoef(
            predictions,
            targets
        )[0, 1]

    return float(mae), float(rmse), float(pearson)


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images).squeeze(1)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_dataset)


@torch.no_grad()
def evaluate(loader):

    model.eval()

    predictions = []
    targets_all = []

    running_loss = 0.0

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images).squeeze(1)

        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        predictions.extend(
            outputs.detach().cpu().numpy()
        )

        targets_all.extend(
            targets.detach().cpu().numpy()
        )

    loss = running_loss / len(loader.dataset)

    mae, rmse, pearson = calculate_metrics(
        predictions,
        targets_all
    )

    return loss, mae, rmse, pearson


# ============================================================
# Training
# ============================================================

print("=" * 60)
print("Starting Training")
print("=" * 60)

best_val_rmse = float("inf")

history = []

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()

    val_loss, val_mae, val_rmse, val_pearson = evaluate(
        val_loader
    )

    scheduler.step(val_rmse)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.4f} | "
        f"Val RMSE: {val_rmse:.4f} | "
        f"Val Pearson: {val_pearson:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_pearson": val_pearson,
        "learning_rate": current_lr
    })

    if val_rmse < best_val_rmse:

        best_val_rmse = val_rmse

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_pearson": val_pearson
            },
            MODEL_PATH
        )

        print("  -> Best model saved.")


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# Load best model
# ============================================================

print()
print("Loading best model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Final evaluation
# ============================================================

print()
print("=" * 60)
print("Final Evaluation")
print("=" * 60)

_, val_mae, val_rmse, val_pearson = evaluate(
    val_loader
)

_, test_mae, test_rmse, test_pearson = evaluate(
    test_loader
)


print()
print("Validation")
print("-" * 40)
print(f"MAE                 : {val_mae:.4f}")
print(f"RMSE                : {val_rmse:.4f}")
print(f"Pearson Correlation : {val_pearson:.4f}")

print()
print("Test")
print("-" * 40)
print(f"MAE                 : {test_mae:.4f}")
print(f"RMSE                : {test_rmse:.4f}")
print(f"Pearson Correlation : {test_pearson:.4f}")


# ============================================================
# Save results
# ============================================================

results = {
    "model": "ResNet18",
    "device": str(DEVICE),
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),
    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "learning_rate": LEARNING_RATE,
    "best_validation_rmse": best_val_rmse,
    "validation": {
        "MAE": val_mae,
        "RMSE": val_rmse,
        "Pearson": val_pearson
    },
    "test": {
        "MAE": test_mae,
        "RMSE": test_rmse,
        "Pearson": test_pearson
    }
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# Finished
# ============================================================

print()
print("=" * 60)
print("Training Completed")
print("=" * 60)

print(f"Model:")
print(MODEL_PATH)

print()
print(f"Training history:")
print(HISTORY_PATH)

print()
print(f"Results:")
print(RESULTS_PATH)

print("=" * 60)