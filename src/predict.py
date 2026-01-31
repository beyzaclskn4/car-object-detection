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

def predict_bbox(model, image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        bbox = model(image_tensor)[0].cpu().numpy()

    x1, y1, x2, y2 = bbox

    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    return (x1, y1, x2, y2), image
