import torch
import torch.nn as nn
from torchvision import models


class BoundingBoxModel(nn.Module):
    """
    CNN-based regression model for bounding box prediction.
    Output: [x_min, y_min, x_max, y_max]
    """
    