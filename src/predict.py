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

def load_model():
    model = get_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model
