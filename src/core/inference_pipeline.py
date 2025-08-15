# FILE: src/core/inference_pipeline.py
import torch
import numpy as np
from typing import Optional
from pathlib import Path
import sys

# Ensure the project root is in the Python path
try:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except (IndexError, NameError):
    ROOT = Path.cwd()

from src.models.unet_segmenter import UNet

class RegenerationInferencePipeline:
    """
    Handles the loading of a trained U-Net model and performs inference
    on multi-spectral data patches to generate regeneration suitability maps.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initializes the pipeline by loading the trained model.

        Args:
            model_path (str): Path to the saved .pth file for the U-Net model.
            device (str): The device to run inference on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ U-Net model loaded successfully from {model_path} onto {device}.")
        else:
            print(f"❌ Failed to load U-Net model from {model_path}.")


    def _load_model(self, model_path: str) -> Optional[UNet]:
        """Safely loads the U-Net model from a file."""
        try:
            # The UNet.load classmethod handles loading init_kwargs and state_dict
            # We explicitly pass the expected in_channels to ensure compatibility
            model = UNet.load(model_path, device=self.device, in_channels=15)
            return model
        except FileNotFoundError:
            print(f"Warning: Model file not found at {model_path}. Inference will be disabled.")
            return None
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def predict(self, patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Performs semantic segmentation on a multi-spectral patch.

        Args:
            patch (np.ndarray): A NumPy array of shape (Channels, Height, Width).

        Returns:
            Optional[np.ndarray]: A 2D NumPy array representing the segmentation mask,
                                  or None if the model is not loaded.
        """
        if self.model is None:
            return None
            
        with torch.no_grad():
            # Convert NumPy array to PyTorch tensor and add a batch dimension
            input_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)
            
            # Perform inference
            logits = self.model(input_tensor)
            
            # Get the predicted class for each pixel
            prediction = torch.argmax(logits, dim=1)
            
            # Move result to CPU and convert to NumPy array, removing the batch dimension
            mask = prediction.squeeze(0).cpu().numpy()
            
            return mask

if __name__ == '__main__':
    # --- Smoke Test for the Inference Pipeline ---
    print("--- Initializing Inference Pipeline ---")
    
    # Create a dummy model file for testing purposes
    IN_CHANNELS = 15
    NUM_CLASSES = 5
    DUMMY_MODEL_PATH = "dummy_unet_model.pth"
    
    dummy_model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    dummy_model.save(DUMMY_MODEL_PATH)
    print(f"Created dummy model at {DUMMY_MODEL_PATH}")
    
    # Initialize the pipeline
    pipeline = RegenerationInferencePipeline(model_path=DUMMY_MODEL_PATH)
    
    if pipeline.model:
        # Create a synthetic multi-spectral patch
        PATCH_SIZE = 256
        synthetic_patch = np.random.rand(IN_CHANNELS, PATCH_SIZE, PATCH_SIZE).astype(np.float32)
        print(f"\n--- Performing prediction on synthetic patch of shape {synthetic_patch.shape} ---")
        
        segmentation_mask = pipeline.predict(synthetic_patch)
        
        if segmentation_mask is not None:
            print("✅ Prediction successful.")
            print(f"   - Output mask shape: {segmentation_mask.shape}")
            
            expected_shape = (PATCH_SIZE, PATCH_SIZE)
            assert segmentation_mask.shape == expected_shape, "Mask shape is incorrect!"
            print("✅ Output mask shape is correct.")
            
            assert segmentation_mask.min() >= 0 and segmentation_mask.max() < NUM_CLASSES
            print("✅ Mask class values are within the expected range.")
            
        else:
            print("❌ Prediction failed.")
    
    # Clean up the dummy model file
    import os
    if os.path.exists(DUMMY_MODEL_PATH):
        os.remove(DUMMY_MODEL_PATH)
        print(f"\nCleaned up {DUMMY_MODEL_PATH}.")
        
    print("\n--- Smoke Test Finished ---")
