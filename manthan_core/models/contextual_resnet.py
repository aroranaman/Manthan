import torch
import torch.nn as nn
import torchvision.models as models

class LandUseClassifier(nn.Module):
    """
    A ResNet-50 based model focused exclusively on land-use classification.
    """
    def __init__(self, num_landuse_classes: int):
        super().__init__()
        # Load a pre-trained ResNet-50 model
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        # Replace the final layer with a new one for our specific task
        self.backbone.fc = nn.Linear(num_ftrs, num_landuse_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)