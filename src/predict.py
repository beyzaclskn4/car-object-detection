import os
import torch
import cv2
from torchvision import transforms

from model import get_model
import os

# ========== DEVICE ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "experiments", "best_model.pth")

print("MODEL PATH:", MODEL_PATH)
# ========== IMAGE ==========
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_model():
    model = get_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict_bbox(model, image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(image_tensor)
        print("RAW MODEL OUTPUT:", preds)

    return preds, image
def predict_bbox(model, image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found! Path yanlış.")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(image_tensor)
        print("RAW MODEL OUTPUT:", preds)

    # 🔽 tensor → numpy → flatten
    bbox = preds.squeeze().cpu().numpy()

    x1, y1, x2, y2 = map(int, bbox)


    return (x1, y1, x2, y2), image


def draw_bbox(image, bbox):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    model = load_model()

    bbox, image = predict_bbox(model, "test.jpg")
    result = draw_bbox(image, bbox)

    cv2.imshow("Prediction", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
