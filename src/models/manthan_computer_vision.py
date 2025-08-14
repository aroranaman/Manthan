# src/models/manthan_computer_vision.py

import torch
from .resnet_classifier import ResNetClassifier
from .unet_segmenter import UNetSegmenter

# Load trained models (update paths accordingly)
resnet_model_path = 'src/trained_models/forest_resnet.pth'
unet_model_path = 'src/trained_models/forest_unet.pth'

# Initialize models
resnet_model = ResNetClassifier()
resnet_model.load_state_dict(torch.load(resnet_model_path))
resnet_model.eval()

unet_model = UNetSegmenter()
unet_model.load_state_dict(torch.load(unet_model_path))
unet_model.eval()

def classify_forest(image_tensor):
    # image_tensor shape: [batch, channels, height, width]
    with torch.no_grad():
        preds = resnet_model(image_tensor)
        probs = torch.softmax(preds, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
    return top_class.cpu().numpy(), top_prob.cpu().numpy()

def segment_forest(image_tensor):
    with torch.no_grad():
        preds = unet_model(image_tensor)
        seg_map = torch.nn.functional.softmax(preds, dim=1)
        predicted_mask = torch.argmax(seg_map, dim=1)
    return predicted_mask.cpu().numpy()
