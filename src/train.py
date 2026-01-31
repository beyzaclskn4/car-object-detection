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
