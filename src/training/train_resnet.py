# FILE: src/training/train_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import sys
import asyncio
from typing import Tuple, List, Optional, Union
from pathlib import Path

# Ensure the project root is in the Python path to locate other modules
try:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except (IndexError, NameError):
    # Fallback for interactive environments
    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.data.real_data_pipeline import get_real_environmental_data
from src.models.resnet_classifier import build_resnet_veg, save as save_model

# --- NEW: Custom Transform for Multi-channel Data ---
class MultiChannelColorJitter:
    """
    A custom transform that applies ColorJitter only to the first 3 (RGB) channels
    of a multi-channel tensor.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] < 3:
            # Not enough channels to apply ColorJitter
            return tensor
            
        # Separate RGB channels from other data channels
        rgb_channels = tensor[:3, :, :]
        other_channels = tensor[3:, :, :]
        
        # Apply ColorJitter to RGB channels
        rgb_channels_transformed = self.color_jitter(rgb_channels)
        
        # Concatenate the transformed RGB with the original other channels
        return torch.cat([rgb_channels_transformed, other_channels], dim=0)


class VegetationPatchDataset(Dataset):
    """
    PyTorch Dataset for loading multi-spectral vegetation patches for classification.
    """
    def __init__(
        self,
        locations: List[Tuple[float, float]],
        num_classes: int,
        num_input_channels: int = 15,
        patch_size: int = 224,
        is_train: bool = True
    ):
        """
        Initializes the dataset.
        """
        self.locations = locations
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.patch_size = patch_size
        
        # CORRECTED: Use the custom transform for multi-channel data
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                MultiChannelColorJitter(brightness=0.1, contrast=0.1), # Use custom jitter
                transforms.Normalize(mean=[0.5] * num_input_channels, std=[0.5] * num_input_channels)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * num_input_channels, std=[0.5] * num_input_channels)
            ])

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.locations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a sample from the dataset.
        """
        lat, lon = self.locations[idx]
        
        image_patch = np.random.rand(self.patch_size, self.patch_size, self.num_input_channels).astype(np.float32)
        
        env_data = asyncio.run(get_real_environmental_data(lat, lon))
        
        if env_data['forest_cover_2021'] > 60 and env_data['canopy_density'] == 'Dense':
            label = 0
        elif env_data['annual_precipitation'] > 2200 and env_data['ndvi_mean'] > 0.7:
            label = 3
        elif env_data['mean_temperature'] > 27 and env_data['annual_precipitation'] < 700:
            label = 2
        elif 15 < env_data['forest_cover_2021'] <= 60:
            label = 1
        else:
            label = 4
            
        image_tensor = self.transform(image_patch)
        
        return image_tensor, label

def train_resnet_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: Union[str, torch.device],
    model_save_path: str
) -> None:
    """
    Training loop for the ResNet vegetation classifier.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    best_val_accuracy = 0.0

    model.to(device)
    
    print("\n--- Starting ResNet Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, model_save_path)
            print(f"✅ Best model saved to {model_save_path} with accuracy: {val_accuracy:.2f}%")

if __name__ == '__main__':
    # --- Configuration ---
    NUM_CLASSES = 5
    NUM_INPUT_CHANNELS = 15
    PATCH_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_PATH = "saved_models/trained_resnet_classifier.pth"
    
    print("--- Running ResNet Classifier Training Script ---")
    print(f"Using device: {DEVICE}")
    
    # --- Data Preparation ---
    train_locations = [
        (10.85, 76.27), (26.91, 75.79), (22.57, 88.36), (28.36, 79.42),
        (12.97, 77.59), (30.31, 78.03), (23.02, 72.57), (17.38, 78.48),
        (25.59, 91.89), (15.29, 73.91), (21.17, 72.83), (32.72, 74.85)
    ]
    val_locations = [
        (11.01, 76.95), (25.31, 82.97), (19.07, 72.87), (31.10, 77.17)
    ]

    train_dataset = VegetationPatchDataset(train_locations, NUM_CLASSES, NUM_INPUT_CHANNELS, PATCH_SIZE)
    val_dataset = VegetationPatchDataset(val_locations, NUM_CLASSES, NUM_INPUT_CHANNELS, PATCH_SIZE, is_train=False)
    
    # Use a single worker for simplicity if issues persist with asyncio in multiprocessing
    num_workers = 0 if sys.platform == "win32" else os.cpu_count() // 2
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # --- Model Initialization ---
    model = build_resnet_veg(
        num_input_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        pretrained=True,
        device=DEVICE,
        seed=42
    )

    # --- Training Execution ---
    train_resnet_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        model_save_path=MODEL_SAVE_PATH
    )

    print("\n--- Training Script Finished ---")
