import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re
import rasterio
import numpy as np

class ManthanDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Manthan's labeled satellite image patches.
    This version is focused solely on land-use classification.
    """
    def __init__(self, image_dir, landuse_map, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff'))]
        self.landuse_map = landuse_map
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.image_files[idx])
        
        try:
            with rasterio.open(img_name) as src:
                img_array = src.read([1, 2, 3]).astype(np.float32)

            if np.isnan(img_array).all() or np.all(img_array == img_array.flat[0]):
                return None

            min_val, max_val = np.nanmin(img_array), np.nanmax(img_array)
            if max_val > min_val:
                img_array = (img_array - min_val) / (max_val - min_val)
            else:
                img_array.fill(0)

            img_array = (img_array * 255).astype(np.uint8)
            img_array = np.transpose(img_array, (1, 2, 0))
            image = Image.fromarray(img_array)

        except Exception as e:
            return None

        filename_body = os.path.basename(img_name)
        # Inside the __getitem__ method, after reading the image with rasterio

        # Check for invalid data (all NaN or a single solid color)
        if np.isnan(img_array).all() or np.all(img_array == img_array.flat[0]):
            print(f"WARNING: Image {img_name} contains invalid data. Skipping.")
            return None # Return None to be filtered out by the collator

        # Safely normalize, handling any remaining NaNs
        min_val, max_val = np.nanmin(img_array), np.nanmax(img_array)
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val)
        else: # If the image is a solid color after all
            img_array.fill(0)

        img_array = (img_array * 255).astype(np.uint8)

        lulc_match = re.search(r"_lulc_([a-zA-Z_]+)\.", filename_body)
        lulc_name = lulc_match.group(1) if lulc_match else "Unknown"
        landuse_label = self.landuse_map.get(lulc_name, -1)
        
        if landuse_label == -1:
            return None

        if self.transform:
            image = self.transform(image)

        return image, landuse_label

def get_transforms(is_train=True):
    """Returns a set of standard transformations for training a ResNet model."""
    if is_train:
        # Apply augmentations only to the training data
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Add random horizontal flips
            transforms.RandomVerticalFlip(),   # Add random vertical flips
            transforms.RandomRotation(20),     # Add random rotations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # No augmentations for the validation/test data
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # In manthan_core/data/image_dataset.py
    def get_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Add random flips
            transforms.RandomRotation(15),     # Add random rotations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])