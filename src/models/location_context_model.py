# FILE: src/models/location_context_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class LocationLandUseAI(nn.Module):
    """
    A multi-head CNN for predicting district and land use from a satellite image patch.
    """
    def __init__(self, num_input_channels: int, num_districts: int, num_land_use_classes: int = 3):
        super().__init__()
        
        # Load a pre-trained ResNet-18 and adapt its first layer for our number of input bands
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        backbone_out_features = resnet.fc.in_features
        
        self.location_head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out_features),
            nn.Dropout(0.5),
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_districts)
        )
        
        self.land_use_head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out_features),
            nn.Dropout(0.5),
            nn.Linear(backbone_out_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_land_use_classes)
        )

    def forward(self, image_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image_patch)
        location_logits = self.location_head(features)
        land_use_logits = self.land_use_head(features)
        return location_logits, land_use_logits