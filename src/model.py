import torch
import torch.nn as nn
from torchvision import models


class BoundingBoxModel(nn.Module):
    """
    CNN-based regression model for bounding box prediction.
    Output: [x_min, y_min, x_max, y_max]
    """
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        return self.backbone(x)
    
def get_model(device):
    model = BoundingBoxModel()
    return model.to(device)