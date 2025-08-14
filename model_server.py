# model_server.py - Create this file
from your_resnet_model import ResNetForestClassifier
from your_unet_model import UNetForestSegmenter  
from your_nfc_species import NFCSpeciesPredictor

class ManthanModelServer:
    def __init__(self):
        self.resnet = ResNetForestClassifier.load_pretrained()
        self.unet = UNetForestSegmenter.load_pretrained()
        self.nfc = NFCSpeciesPredictor.load_pretrained()
    
    def predict_forest_analysis(self, satellite_image_path, lat, lon):
        # ResNet classification
        forest_type = self.resnet.classify(satellite_image_path)
        # UNet segmentation  
        forest_mask = self.unet.segment(satellite_image_path)
        # Species prediction
        species_recommendations = self.nfc.predict_species(lat, lon, forest_type)
        
        return {
            'forest_type': forest_type,
            'forest_boundaries': forest_mask,
            'species_recommendations': species_recommendations
        }
