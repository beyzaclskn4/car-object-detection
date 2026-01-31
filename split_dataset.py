import os
import shutil
import random
import pandas as pd

# --------- AYARLAR ----------
IMAGE_DIR = "training_images"
LABELS_CSV = "labels.csv"
OUTPUT_DIR = "data"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# ----------------------------

random.seed(RANDOM_SEED)

# klasörleri oluştur
os.makedirs(f"{OUTPUT_DIR}/train/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val/images", exist_ok=True)

# tüm resimleri al
images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(".jpg")]
random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

# resimleri taşı
for img in train_images:
    shutil.copy(
        os.path.join(IMAGE_DIR, img),
        f"{OUTPUT_DIR}/train/images/{img}"
    )

for img in val_images:
    shutil.copy(
        os.path.join(IMAGE_DIR, img),
        f"{OUTPUT_DIR}/val/images/{img}"
    )

# CSV'yi böl
df = pd.read_csv(LABELS_CSV)

train_df = df[df["image"].isin(train_images)]
val_df = df[df["image"].isin(val_images)]

train_df.to_csv(f"{OUTPUT_DIR}/train/labels.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val/labels.csv", index=False)

print("✅ Train / Validation split tamamlandı.")
