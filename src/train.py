import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from dataset import CarDataset
import os
from model import get_model


# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "../data/train/labels.csv"
VAL_CSV   = "../data/val/labels.csv"

TRAIN_IMG_DIR = "../data/train/images"
VAL_IMG_DIR   = "../data/val/images"

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

# ========== DATASET & DATALOADER ==========
train_dataset = CarDataset(
    images_dir=TRAIN_IMG_DIR,
    labels_csv=TRAIN_CSV,
    transform=transform
)

val_dataset = CarDataset(
    images_dir=VAL_IMG_DIR,
    labels_csv=VAL_CSV,
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

model = get_model(DEVICE)

# ========== LOSS & OPTIMIZER ==========
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ========== TRAIN & VALIDATION ==========
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    train_loss = 0.0

    for images, bboxes in train_loader:
        images = images.to(DEVICE)
        bboxes = bboxes.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, bboxes)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, bboxes in val_loader:
            images = images.to(DEVICE)
            bboxes = bboxes.to(DEVICE)


            outputs = model(images)
            loss = criterion(outputs, bboxes)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    # ---- SAVE BEST MODEL ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved")
