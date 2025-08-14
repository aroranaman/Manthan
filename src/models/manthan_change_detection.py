# src/models/manthan_change_detection.py

import torch
from .siamese_detector import SiameseNetwork

# Load trained Siamese model
siamese_model_path = 'src/trained_models/siamese_change.pth'
siamese_model = SiameseNetwork()
siamese_model.load_state_dict(torch.load(siamese_model_path))
siamese_model.eval()

def detect_change(image1, image2):
    # image1, image2: tensors
    with torch.no_grad():
        change_score = siamese_model(image1, image2)
    return change_score.cpu().numpy()
