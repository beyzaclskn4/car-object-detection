import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from dataset import CarDataset
import os

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from dataset import CarDataset
import os

# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "data/train/labels.csv"
VAL_CSV = "data/val/labels.csv"

TRAIN_IMG_DIR = "data/train/images"
VAL_IMG_DIR = "data/val/images"

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

MODEL_SAVE_PATH = "experiments/best_model.pth"
os.makedirs("experiments", exist_ok=True)

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ========== DATASETS & DATALOADERS ==========
# ========== DATASET & DATALOADER ==========
train_dataset = CarDataset(
    csv_path=TRAIN_CSV,
    img_dir=TRAIN_IMG_DIR,
    transform=transform
)

val_dataset = CarDataset(
    csv_path=VAL_CSV,
    img_dir=VAL_IMG_DIR,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

