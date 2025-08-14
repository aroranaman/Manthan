# src/models/siamese_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.encoder(x)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = BaseCNN()

    def forward(self, x1, x2):
        feat1 = self.cnn(x1)
        feat2 = self.cnn(x2)
        distance = F.cosine_similarity(feat1, feat2)
        return distance
