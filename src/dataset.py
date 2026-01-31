import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CarDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        self.data = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.images_dir, row["image"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # bbox: xmin, ymin, xmax, ymax
        bbox = torch.tensor(
            [row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
            dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, bbox
