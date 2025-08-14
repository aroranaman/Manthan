# FILE: src/training/train_unet.py
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Tuple, List, Optional, Union
from pathlib import Path

# Ensure the project root is in the Python path to locate other modules
try:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except (IndexError, NameError):
    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.models.unet_segmenter import build_unet_segmenter, UNet

class LandCoverDataset(Dataset):
    """
    PyTorch Dataset for loading multi-spectral image patches and segmentation masks.

    This dataset simulates the generation of synthetic multi-spectral image patches
    and corresponding land cover segmentation masks for training a UNet model.
    """
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        num_input_channels: int = 4,
        patch_size: int = 256
    ):
        """
        Initializes the dataset.

        Args:
            num_samples (int): The number of synthetic samples to generate.
            num_classes (int): The number of land cover classes for the masks.
            num_input_channels (int): The number of channels for the synthetic image.
            patch_size (int): The height and width of the image and mask.
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.patch_size = patch_size

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and the mask tensor.
        """
        # --- Synthetic Data Generation ---
        image = np.random.rand(self.num_input_channels, self.patch_size, self.patch_size).astype(np.float32)
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)
        
        # Generate a more complex, realistic mask
        num_features = np.random.randint(3, 7)
        for _ in range(num_features):
            class_id = np.random.randint(1, self.num_classes)
            feature_type = np.random.choice(['circle', 'rectangle', 'strip'])
            
            if feature_type == 'circle':
                cx, cy = np.random.randint(0, self.patch_size, 2)
                r = np.random.randint(10, self.patch_size // 4)
                y, x = np.ogrid[-cy:self.patch_size-cy, -cx:self.patch_size-cx]
                mask[x*x + y*y <= r*r] = class_id
            
            elif feature_type == 'rectangle':
                x1, y1 = np.random.randint(0, self.patch_size - 20, 2)
                w, h = np.random.randint(20, self.patch_size // 3, 2)
                x2, y2 = min(x1 + w, self.patch_size), min(y1 + h, self.patch_size)
                mask[y1:y2, x1:x2] = class_id
                
            elif feature_type == 'strip':
                if np.random.rand() > 0.5: # Horizontal
                    y1 = np.random.randint(0, self.patch_size - 10)
                    h = np.random.randint(5, 15)
                    mask[y1:y1+h, :] = class_id
                else: # Vertical
                    x1 = np.random.randint(0, self.patch_size - 10)
                    w = np.random.randint(5, 15)
                    mask[:, x1:x1+w] = class_id

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        
        return image_tensor, mask_tensor

def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6):
    """
    Calculates the multiclass Dice loss for semantic segmentation.
    """
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = torch.sum(pred * target_one_hot, dims=(2, 3))
    union = torch.sum(pred, dims=(2, 3)) + torch.sum(target_one_hot, dims=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice.mean()

def train_unet_segmenter(
    model: UNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: Union[str, torch.device],
    model_save_path: str,
    num_classes: int
) -> None:
    """
    Training loop for the UNet land cover segmenter.

    Args:
        model (UNet): The UNet model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (Union[str, torch.device]): The device to train on ('cpu' or 'cuda').
        model_save_path (str): Path to save the best model.
        num_classes (int): The number of segmentation classes.
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    model.to(device)
    
    print("\n--- Starting UNet Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = dice_loss(outputs, masks, num_classes)
            loss = 0.5 * loss_ce + 0.5 * loss_dice  # Combine losses
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = dice_loss(outputs, masks, num_classes)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(model_save_path)
            print(f"✅ Best model saved to {model_save_path} with validation loss: {avg_val_loss:.4f}")

if __name__ == '__main__':
    # --- Configuration ---
    NUM_CLASSES = 5  # e.g., Water, Urban, Forest, Agriculture, Barren
    NUM_INPUT_CHANNELS = 4 # e.g., RGB + NIR
    PATCH_SIZE = 256
    BATCH_SIZE = 4
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_PATH = "trained_unet_segmenter.pth"
    
    print("--- Running UNet Segmenter Training Script ---")
    print(f"Using device: {DEVICE}")
    
    # --- Data Preparation ---
    train_dataset = LandCoverDataset(num_samples=32, num_classes=NUM_CLASSES, num_input_channels=NUM_INPUT_CHANNELS, patch_size=PATCH_SIZE)
    val_dataset = LandCoverDataset(num_samples=16, num_classes=NUM_CLASSES, num_input_channels=NUM_INPUT_CHANNELS, patch_size=PATCH_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2)

    # --- Model Initialization ---
    model = build_unet_segmenter(
        in_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        use_attention=True,
        device=DEVICE,
        seed=42
    )

    # --- Training Execution ---
    train_unet_segmenter(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        model_save_path=MODEL_SAVE_PATH,
        num_classes=NUM_CLASSES
    )

    print("\n--- Training Script Finished ---")
