import os
import pandas as pd
from PIL import Image #görsel okuma için

import torch
from torch.utils.data import Dataset

class CarDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform

        self.image_ids = self.labels["image"].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)

        image = Image.open(image_path).convert("RGB")

        records = self.labels[self.labels["image"] == image_id]

        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes
        }

        if self.transform:
            image = self.transform(image)

        return image, target

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

if __name__ == "__main__":
    dataset = CarDataset(
        images_dir="data/train/images",
        labels_csv="data/train/labels.csv"
    )

    img, target = dataset[0]
    print(img.size)
    print(target)
    print("Dataset test tamamlandı.")