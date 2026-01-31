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
