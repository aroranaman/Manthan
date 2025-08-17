# FILE: src/data/geo_dataset.py
import torch
import pandas as pd
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class ManthanGeoDataset(Dataset):
    """
    PyTorch dataset for loading satellite image patches and their labels.
    Handles both multi-band GeoTIFFs and standard image formats like PNG.
    """
    def __init__(self, manifest_path: Path, image_dir: Path, transform=None):
        self.df = pd.read_csv(manifest_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['image_filename']
        
        # --- CORRECTED: Flexible image loading ---
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            # Load multi-band GeoTIFF file
            with rasterio.open(img_path) as src:
                # Ensure we read a maximum of 4 bands for consistency
                image = src.read(list(range(1, min(src.count, 4) + 1))).astype('float32')
        else:
            # Load standard image formats like PNG, JPG
            with Image.open(img_path) as img:
                # Ensure it has 4 channels (e.g., RGBA)
                img = img.convert("RGBA")
                # Convert PIL image to numpy array and transpose to (C, H, W)
                image = np.array(img, dtype='float32').transpose((2, 0, 1))

        # Normalize 0-255 images to 0-1 range before transformations
        if image.max() > 1.0:
            image = image / 255.0

        district_label = int(row['district_id'])
        land_use_label = int(row['land_use_id'])
        
        # The transform pipeline expects a tensor
        image_tensor = torch.from_numpy(image)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, district_label, land_use_label