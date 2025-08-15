# FILE: src/core/manthan_ai_engine.py
# The final, unified AI engine that orchestrates all models.

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any

# --- Robust Path Fix ---
import sys
from pathlib import Path
try:
    project_root = Path(__file__).resolve().parents[2]
except IndexError:
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Fix ---

from src.core.geospatial_handler import GeospatialDataHandler
from src.core.inference_pipeline import RegenerationInferencePipeline
from src.models.unet_segmenter import UNet
from src.models.resnet_classifier import build_resnet_veg
from src.models.deep_survival_predictor import DeepSurvivalPredictor
from src.core.knowledge_graph import EcologicalKnowledgeGraph
from src.core.economic_viability_model import EconomicViabilityModel
from src.core.market_linkage_model import MarketLinkageModel

class ManthanAIEngine:
    """
    The master AI engine that integrates all data and model pipelines
    to generate a comprehensive land restoration report.
    """
    def __init__(self, model_dir="saved_models"):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize all components
        self.geospatial_handler = GeospatialDataHandler()
        self.unet_pipeline = self._initialize_unet()
        self.resnet_classifier = self._initialize_resnet()
        self.survival_predictor = self._initialize_survival_predictor()
        self.knowledge_graph = EcologicalKnowledgeGraph()
        self.economic_model = EconomicViabilityModel()
        self.market_model = MarketLinkageModel()
        
        print("✅ Manthan AI Engine initialized with all models.")

    def _initialize_unet(self):
        path = self.model_dir / "trained_unet_segmenter.pth"
        if not path.exists():
            print(f"⚠️ U-Net model not found. Creating a dummy model.")
            dummy_model = UNet(in_channels=15, num_classes=5)
            dummy_model.save(str(path))
        return RegenerationInferencePipeline(model_path=str(path), device=self.device)

    def _initialize_resnet(self):
        path = self.model_dir / "trained_resnet_classifier.pth"
        # In a real scenario, you would load a trained model. Here we initialize a new one.
        model = build_resnet_veg(num_input_channels=15, num_classes=5, pretrained=True, device=self.device)
        return model

    def _initialize_survival_predictor(self):
        path = self.model_dir / "trained_survival_predictor.pth"
        model = DeepSurvivalPredictor(num_species=15000)
        # In a real scenario, load trained weights.
        # model.load_state_dict(torch.load(path))
        return model.to(self.device)

    def run_full_analysis(self, aoi_geojson: Dict) -> Dict[str, Any]:
        """
        Executes the entire AI pipeline from data ingestion to final report generation.
        """
        # 1. Data Ingestion
        patch, error = self.geospatial_handler.get_multispectral_patch(aoi_geojson)
        if error:
            raise RuntimeError(f"Data Ingestion Failed: {error}")
        patch_tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).to(self.device)

        # 2. U-Net Land Suitability
        suitability_map = self.unet_pipeline.predict(patch)
        
        # 3. ResNet Land Classification
        with torch.no_grad():
            # ResNet needs a different input size
            resnet_input = torch.nn.functional.interpolate(patch_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            land_class_logits = self.resnet_classifier(resnet_input)
            land_class_pred = torch.argmax(land_class_logits, dim=1).item()
        
        # 4. Deep Survival & Knowledge Graph for Species
        env_vector = self._create_feature_vector(patch)
        species_recommendations = self._get_species_recommendations(env_vector)
        
        # 5. Economic & Market Analysis
        area_ha = self._calculate_area(aoi_geojson)
        suitability_score = np.sum(suitability_map >= 3) / suitability_map.size
        economic_projections = self.economic_model.get_projections(area_ha, suitability_score)
        market_analysis = self.market_model.match_farmer_to_buyers(
            [rec['name'] for rec in species_recommendations['primary_species']]
        )

        return {
            "suitability_map": suitability_map,
            "land_classification": ["Degraded Land", "Scrubland", "Open Forest", "Cropland", "Dense Forest"][land_class_pred],
            "species_recommendations": species_recommendations,
            "economic_projections": economic_projections,
            "market_analysis": market_analysis.to_dict(orient='records')
        }

    def _create_feature_vector(self, patch: np.ndarray) -> torch.Tensor:
        """Creates the 47-dimensional feature vector for the survival predictor."""
        means = patch.mean(axis=(1, 2))
        stds = patch.std(axis=(1, 2))
        mins = patch.min(axis=(1, 2))
        maxs = patch.max(axis=(1, 2))
        # This creates a 15*4 = 60 feature vector. We'll use the first 47.
        full_vector = np.concatenate([means, stds, mins, maxs]).astype(np.float32)
        return torch.from_numpy(full_vector[:47]).to(self.device)

    def _get_species_recommendations(self, env_vector: torch.Tensor) -> Dict:
        """Generates species recommendations using the deep learning model and knowledge graph."""
        self.survival_predictor.eval()
        with torch.no_grad():
            all_species_ids = torch.arange(self.survival_predictor.num_species, device=self.device)
            env_vector_batch = env_vector.unsqueeze(0).repeat(self.survival_predictor.num_species, 1)
            survival_probs = self.survival_predictor(all_species_ids, env_vector_batch).squeeze()
            
            top_probs, top_indices = torch.topk(survival_probs, 10)
        
        # Dummy names for now
        species_names = {i: f"Native Species #{i}" for i in top_indices.cpu().numpy()}
        
        recommendations = [
            {"name": species_names[idx.item()], "survival_probability": prob.item()}
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        primary = recommendations[:5]
        secondary = recommendations[5:]
        
        # Enhance with Knowledge Graph
        primary_names = [p['name'] for p in primary]
        companions = self.knowledge_graph.get_symbiotic_partners("Shorea robusta") # Demo
        warnings = self.knowledge_graph.check_invasive_interactions(primary_names)
        
        return {
            "primary_species": primary,
            "secondary_species": secondary,
            "companion_plants": companions,
            "invasive_warnings": warnings
        }

    def _calculate_area(self, aoi: Dict) -> float:
        """Calculates the approximate area of the AOI in hectares."""
        try:
            coords = aoi['geometry']['coordinates'][0]
            lon_range = max(c[0] for c in coords) - min(c[0] for c in coords)
            lat_range = max(c[1] for c in coords) - min(c[1] for c in coords)
            area_ha = (lon_range * 111) * (lat_range * 111) * 100
            return max(area_ha, 0.1)
        except:
            return 1.0
