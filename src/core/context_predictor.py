# FILE: src/core/context_predictor.py
"""
ContextPredictor: load trained LocationLandUseAI and run CPU inference.

Fixes:
- Robust path setup so running this file directly works.
- Passes num_land_use_classes to the model.
- Accepts numpy patches shaped (H,W,4) or (4,H,W).
- Clear errors if model / mapping are missing.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch
from torchvision import transforms

# --- Robust Path Fix (allows `python src/core/context_predictor.py`) ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except Exception:
    PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ----------------------------------------------------------------------

from src.models.location_context_model import LocationLandUseAI  # noqa: E402


class ContextPredictor:
    """Loads and runs the trained LocationLandUseAI model for inference."""

    def __init__(
        self,
        model_path: Path,
        mapping_path: Path,
        num_input_channels: int = 4,
        num_land_use_classes: int = 3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"District mapping not found: {mapping_path}\n"
                "Expected a JSON like {'District Name': id, ...}"
            )
        with open(mapping_path, "r") as f:
            # stored as {name: id} → invert to {id: name}
            name_to_id = json.load(f)
        self.district_id_to_name = {v: k for k, v in name_to_id.items()}

        self.land_use_id_to_name = {
            0: "Urban/Other",
            1: "Rural (Cropland)",
            2: "Forested",
        }

        num_districts = max(1, len(self.district_id_to_name))

        self.model = LocationLandUseAI(
            num_input_channels=num_input_channels,
            num_districts=num_districts,
            num_land_use_classes=num_land_use_classes,
        ).to(self.device)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Train it or point to a valid .pth"
            )

        state = torch.load(model_path, map_location=self.device)
        # Accept either full checkpoint or just state_dict
        if isinstance(state, dict) and "state_dict" in state:
            self.model.load_state_dict(state["state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

        # If your dataset normalized to mean/std below, keep the same here.
        # ConvertImageDtype scales uint8→float32 in [0,1] automatically.
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.3, 0.3, 0.3, 0.3],
                                 std=[0.2, 0.2, 0.2, 0.2]),
        ])

    def _ensure_chw4(self, arr: np.ndarray) -> np.ndarray:
        """Ensure (C,H,W) with 4 channels."""
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr.shape}")
        # (H,W,4) → (4,H,W)
        if arr.shape[-1] == 4:
            arr = np.transpose(arr, (2, 0, 1))
        # already (4,H,W)
        if arr.shape[0] != 4:
            raise ValueError(
                f"Expected 4 channels, got shape {arr.shape}. "
                "Make sure patch has 4 bands (e.g., B2,B3,B4,B8 or RGBA)."
            )
        return arr

    @torch.no_grad()
    def predict(self, image_patch_numpy: np.ndarray) -> Dict[str, str]:
        """Predicts district + land use from a NumPy image patch.

        Accepts (H,W,4) or (4,H,W); dtype uint8/float32 are fine.
        """
        chw = self._ensure_chw4(image_patch_numpy).copy()

        tensor = torch.from_numpy(chw)
        # If float is 0..255, scale roughly to 0..1 before normalize
        if tensor.dtype.is_floating_point and tensor.max() > 1.5:
            tensor = tensor / 255.0

        tensor = self.transform(tensor).unsqueeze(0).to(self.device)

        loc_logits, lu_logits = self.model(tensor)
        pred_district_id = int(torch.argmax(loc_logits, dim=1).item())
        pred_land_use_id = int(torch.argmax(lu_logits, dim=1).item())

        district_name = self.district_id_to_name.get(pred_district_id, "Unknown District")
        land_use_name = self.land_use_id_to_name.get(pred_land_use_id, "Unknown")

        return {
            "district": district_name,
            "area_type": land_use_name,
            "district_id": pred_district_id,
            "land_use_id": pred_land_use_id,
        }


# ----------------- Smoke test -----------------
if __name__ == "__main__":
    # Defaults to your repo layout:
    default_model = PROJECT_ROOT / "saved_models" / "location_context_model.pth"
    default_mapping = PROJECT_ROOT / "src" / "data" / "district_mapping.json"

    print(">>> ContextPredictor smoke test")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path:   {default_model}")
    print(f"Mapping path: {default_mapping}")

    # Create a dummy 4-band patch (64x64) just to exercise the code path.
    dummy_patch = (np.random.rand(64, 64, 4) * 255).astype(np.uint8)

    try:
        predictor = ContextPredictor(
            model_path=default_model,
            mapping_path=default_mapping,
            num_input_channels=4,
            num_land_use_classes=3,
            device="cpu",
        )
        result = predictor.predict(dummy_patch)
        print("Prediction:", result)
    except Exception as e:
        print("❌ Smoke test failed:", e)
        print("Hints:")
        print("  - Ensure the model file exists at saved_models/location_context_model.pth")
        print("  - Ensure the mapping exists at src/data/district_mapping.json")
        print("  - Train with scripts/train_location_model.py if needed.")
