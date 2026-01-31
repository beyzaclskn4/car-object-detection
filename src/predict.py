import torch
import cv2
from torchvision import transforms

from model import get_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/best_model.pth"
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
])
