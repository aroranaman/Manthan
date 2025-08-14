# FILE: src/models/resnet_veg.py
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union

def save(model: nn.Module, path: str) -> None:
    """Saves the model state dictionary to a file.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str): The file path to save the model to.
    """
    torch.save(model.state_dict(), path)

def load(path: str, num_input_channels: int, num_classes: int, device: Union[str, torch.device] = "cpu") -> nn.Module:
    """Loads a ResNet-Veg model from a file.

    Args:
        path (str): The file path to load the model from.
        num_input_channels (int): The number of input channels for the model.
        num_classes (int): The number of output classes for the model.
        device (Union[str, torch.device], optional): The device to load the model onto. Defaults to "cpu".

    Returns:
        nn.Module: The loaded PyTorch model.
    """
    model = build_resnet_veg(num_input_channels=num_input_channels, num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def build_resnet_veg(
    num_input_channels: int,
    num_classes: int,
    pretrained: bool = False,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None
) -> nn.Module:
    """
    Builds a ResNet-50 model adapted for multi-spectral satellite imagery.

    The first convolutional layer is modified to accept the specified number of input channels.
    If pretrained, it inflates the weights from a standard RGB ResNet-50 model.

    Args:
        num_input_channels (int): Number of channels in the input tensor (e.g., 6 for RGB+NIR+SWIR).
        num_classes (int): Number of output classes for the classification task.
        pretrained (bool, optional): If True, loads weights from a ResNet-50 model pretrained on ImageNet. Defaults to False.
        device (Union[str, torch.device], optional): The device to move the model to. Defaults to "cpu".
        seed (Optional[int], optional): A random seed for reproducibility. Defaults to None.

    Returns:
        nn.Module: The modified ResNet-50 model.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Load the ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    # Modify the first convolutional layer to handle multi-spectral input
    if num_input_channels != 3:
        original_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        if pretrained:
            # Inflate weights from the original 3-channel conv layer
            with torch.no_grad():
                original_weights = original_conv1.weight.clone()  # Shape: [64, 3, 7, 7]
                
                # Compute the mean of the RGB weights
                weight_mean = original_weights.mean(dim=1, keepdim=True) # Shape: [64, 1, 7, 7]
                
                # Tile the mean weights to match the new number of input channels
                new_conv1.weight[:, :] = weight_mean.repeat(1, num_input_channels, 1, 1)
                
                # Copy the first 3 channels from the original weights
                copy_channels = min(num_input_channels, 3)
                new_conv1.weight[:, :copy_channels] = original_weights[:, :copy_channels]

        model.conv1 = new_conv1

    # Replace the final fully connected layer for the new classification task
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.to(device)

if __name__ == '__main__':
    print("--- Running ResNet-Veg Smoke Test ---")
    
    # --- Configuration ---
    B, C, H, W = 2, 6, 224, 224
    NUM_CLASSES = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    print(f"Input tensor shape: ({B}, {C}, {H}, {W})")
    print(f"Number of classes: {NUM_CLASSES}")

    # --- Model Creation ---
    # Test both pretrained and non-pretrained initialization
    print("\n1. Building model without pretrained weights...")
    model_scratch = build_resnet_veg(
        num_input_channels=C,
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=DEVICE,
        seed=42
    )
    
    print("\n2. Building model with inflated pretrained weights...")
    model_pretrained = build_resnet_veg(
        num_input_channels=C,
        num_classes=NUM_CLASSES,
        pretrained=True,
        device=DEVICE,
        seed=42
    )

    # --- Forward Pass ---
    print("\n3. Performing forward pass...")
    # Create a synthetic input tensor
    synthetic_input = torch.randn(B, C, H, W, device=DEVICE)
    
    # Get model output
    with torch.no_grad():
        model_scratch.eval()
        logits = model_scratch(synthetic_input)

    # --- Shape Verification ---
    print(f"Input shape: {synthetic_input.shape}")
    print(f"Output (logits) shape: {logits.shape}")
    
    expected_shape = (B, NUM_CLASSES)
    assert logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
    print("✅ Output shape is correct.")

    # --- Save/Load Test ---
    print("\n4. Testing model serialization...")
    model_path = "resnet_veg_test.pth"
    save(model_scratch, model_path)
    print(f"Model saved to {model_path}")
    
    loaded_model = load(model_path, num_input_channels=C, num_classes=NUM_CLASSES, device=DEVICE)
    print("Model loaded successfully.")
    
    # Verify loaded model produces same output
    with torch.no_grad():
        loaded_model.eval()
        loaded_logits = loaded_model(synthetic_input)
    
    assert torch.allclose(logits, loaded_logits, atol=1e-6), "Mismatch between original and loaded model outputs."
    print("✅ Loaded model output matches original model output.")
    
    # Clean up the saved model file
    import os
    os.remove(model_path)
    print(f"Cleaned up {model_path}.")
    
    print("\n--- Smoke Test Passed ---")
```python
# FILE: src/models/unet_seg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Type

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class AttentionGate(nn.Module):
    """Attention Gate for UNet skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        if self.use_attention:
            self.att = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if self.use_attention:
            x2 = self.att(g=x1, x=x2)
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet model for semantic segmentation with optional attention gates.
    """
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, use_attention: bool = True, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.use_attention = use_attention
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, use_attention)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, use_attention)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, use_attention)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, use_attention)
        self.outc = OutConv(base_channels, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes model weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet model."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def save(self, path: str) -> None:
        """Saves the model state dictionary and init args to a file."""
        torch.save({
            'init_kwargs': {
                'in_channels': self.in_channels,
                'num_classes': self.num_classes,
                'base_channels': self.base_channels,
                'use_attention': self.use_attention,
                'bilinear': self.bilinear
            },
            'model_state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls: Type['UNet'], path: str, device: Union[str, torch.device] = "cpu") -> 'UNet':
        """Loads a UNet model from a file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['init_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)


if __name__ == '__main__':
    print("--- Running UNet Smoke Test ---")
    
    # --- Configuration ---
    B, C_IN, H, W = 1, 4, 256, 256
    NUM_CLASSES = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    print(f"Input tensor shape: ({B}, {C_IN}, {H}, {W})")
    print(f"Number of classes: {NUM_CLASSES}")

    # --- Model Creation ---
    print("\n1. Building UNet with Attention...")
    model = UNet(
        in_channels=C_IN,
        num_classes=NUM_CLASSES,
        base_channels=32,
        use_attention=True,
        bilinear=True
    ).to(DEVICE)

    # --- Forward Pass ---
    print("\n2. Performing forward pass...")
    synthetic_input = torch.randn(B, C_IN, H, W, device=DEVICE)
    with torch.no_grad():
        model.eval()
        logits = model(synthetic_input)

    # --- Shape Verification ---
    print(f"Input shape: {synthetic_input.shape}")
    print(f"Output (logits) shape: {logits.shape}")
    
    expected_shape = (B, NUM_CLASSES, H, W)
    assert logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
    print("✅ Output shape is correct.")

    # --- Save/Load Test ---
    print("\n3. Testing model serialization...")
    model_path = "unet_test.pth"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    loaded_model = UNet.load(model_path, device=DEVICE)
    print("Model loaded successfully.")
    
    with torch.no_grad():
        loaded_model.eval()
        loaded_logits = loaded_model(synthetic_input)
    
    assert torch.allclose(logits, loaded_logits, atol=1e-6), "Mismatch between original and loaded model outputs."
    print("✅ Loaded model output matches original model output.")
    
    import os
    os.remove(model_path)
    print(f"Cleaned up {model_path}.")
    
    print("\n--- Smoke Test Passed ---")
```python
# FILE: src/models/ncf_species.py
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Type

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model for predicting species suitability.

    This model combines embeddings for regions and species with dense environmental
    features to produce a suitability score.
    """
    def __init__(
        self,
        num_regions: int,
        num_species: int,
        env_dim: int,
        emb_dim: int = 16,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.1
    ):
        """
        Initializes the NCF model.

        Args:
            num_regions (int): The total number of unique regions.
            num_species (int): The total number of unique species.
            env_dim (int): The dimensionality of the environmental feature vector.
            emb_dim (int, optional): The dimensionality of the region and species embeddings. Defaults to 16.
            hidden_dims (list[int], optional): A list of hidden layer sizes for the MLP. Defaults to [64, 32].
            dropout (float, optional): The dropout rate to apply between MLP layers. Defaults to 0.1.
        """
        super().__init__()
        self.num_regions = num_regions
        self.num_species = num_species
        self.env_dim = env_dim
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Embedding layers
        self.region_embedding = nn.Embedding(num_regions, emb_dim)
        self.species_embedding = nn.Embedding(num_species, emb_dim)

        # MLP layers
        mlp_layers = []
        input_dim = 2 * emb_dim + env_dim
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, region_ids: torch.LongTensor, species_ids: torch.LongTensor, env_feats: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass for a batch of interactions.

        Args:
            region_ids (torch.LongTensor): A tensor of region IDs. Shape: (B,).
            species_ids (torch.LongTensor): A tensor of species IDs. Shape: (B,).
            env_feats (torch.FloatTensor): A tensor of environmental features. Shape: (B, env_dim).

        Returns:
            torch.FloatTensor: A tensor of suitability scores. Shape: (B,).
        """
        region_emb = self.region_embedding(region_ids)
        species_emb = self.species_embedding(species_ids)
        
        # Concatenate embeddings and features
        x = torch.cat([region_emb, species_emb, env_feats], dim=-1)
        
        # Pass through MLP
        logits = self.mlp(x)
        
        # Apply sigmoid to get score
        scores = self.sigmoid(logits).squeeze(-1)
        return scores

    def predict_topk(self, region_id: int, env_feats: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the top-k most suitable species for a given region and environment.

        Args:
            region_id (int): The ID of the region to predict for.
            env_feats (torch.Tensor): The environmental feature tensor for the location. Shape: (env_dim,).
            k (int, optional): The number of top species to return. Defaults to 10.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - topk_indices (torch.Tensor): The indices of the top-k species. Shape: (k,).
                - topk_scores (torch.Tensor): The suitability scores of the top-k species. Shape: (k,).
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Prepare inputs for all species
            all_species_ids = torch.arange(self.num_species, device=device)
            region_ids = torch.tensor([region_id] * self.num_species, device=device)
            env_feats_batch = env_feats.unsqueeze(0).repeat(self.num_species, 1).to(device)
            
            # Get scores for all species
            all_scores = self.forward(region_ids, all_species_ids, env_feats_batch)
            
            # Get top-k scores and indices
            topk_scores, topk_indices = torch.topk(all_scores, k=min(k, self.num_species))
            
        return topk_indices, topk_scores

    def save(self, path: str) -> None:
        """Saves the model state and init args to a file."""
        torch.save({
            'init_kwargs': {
                'num_regions': self.num_regions,
                'num_species': self.num_species,
                'env_dim': self.env_dim,
                'emb_dim': self.emb_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout_rate
            },
            'model_state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls: Type['NCFModel'], path: str, device: Union[str, torch.device] = "cpu") -> 'NCFModel':
        """Loads an NCFModel from a file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['init_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)

def build_ncf_model(
    num_regions: int,
    num_species: int,
    env_dim: int,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    **kwargs
) -> NCFModel:
    """Factory function to build and initialize the NCFModel."""
    if seed is not None:
        torch.manual_seed(seed)
    model = NCFModel(num_regions, num_species, env_dim, **kwargs)
    return model.to(device)

if __name__ == '__main__':
    print("--- Running NCF Smoke Test ---")
    
    # --- Configuration ---
    NUM_REGIONS = 8
    NUM_SPECIES = 20
    ENV_DIM = 5
    BATCH_SIZE = 10
    K_TOP = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    print(f"Num Regions: {NUM_REGIONS}, Num Species: {NUM_SPECIES}, Env Dim: {ENV_DIM}")

    # --- Model Creation ---
    print("\n1. Building NCF model...")
    model = build_ncf_model(
        num_regions=NUM_REGIONS,
        num_species=NUM_SPECIES,
        env_dim=ENV_DIM,
        device=DEVICE,
        seed=42
    )

    # --- Forward Pass Test ---
    print("\n2. Performing forward pass...")
    region_ids = torch.randint(0, NUM_REGIONS, (BATCH_SIZE,), device=DEVICE)
    species_ids = torch.randint(0, NUM_SPECIES, (BATCH_SIZE,), device=DEVICE)
    env_feats = torch.randn(BATCH_SIZE, ENV_DIM, device=DEVICE)
    
    scores = model(region_ids, species_ids, env_feats)
    
    print(f"Input shapes: region_ids={region_ids.shape}, species_ids={species_ids.shape}, env_feats={env_feats.shape}")
    print(f"Output scores shape: {scores.shape}")
    
    expected_shape = (BATCH_SIZE,)
    assert scores.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {scores.shape}"
    assert scores.min() >= 0 and scores.max() <= 1, f"Scores out of [0, 1] range: min={scores.min()}, max={scores.max()}"
    print("✅ Forward pass output shape and range are correct.")

    # --- Top-K Prediction Test ---
    print("\n3. Performing top-k prediction...")
    target_region_id = 3
    target_env_feats = torch.randn(ENV_DIM)
    
    topk_indices, topk_scores = model.predict_topk(target_region_id, target_env_feats, k=K_TOP)
    
    print(f"Top-{K_TOP} indices shape: {topk_indices.shape}")
    print(f"Top-{K_TOP} scores shape: {topk_scores.shape}")
    
    expected_k_shape = (K_TOP,)
    assert topk_indices.shape == expected_k_shape, f"Top-k indices shape mismatch!"
    assert topk_scores.shape == expected_k_shape, f"Top-k scores shape mismatch!"
    print("✅ Top-k prediction output shapes are correct.")
    print("Top-k species indices:", topk_indices.tolist())
    print("Top-k species scores:", [f"{s:.4f}" for s in topk_scores.tolist()])

    # --- Save/Load Test ---
    print("\n4. Testing model serialization...")
    model_path = "ncf_model_test.pth"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    loaded_model = NCFModel.load(model_path, device=DEVICE)
    print("Model loaded successfully.")
    
    loaded_scores = loaded_model(region_ids, species_ids, env_feats)
    assert torch.allclose(scores, loaded_scores, atol=1e-6), "Mismatch between original and loaded model outputs."
    print("✅ Loaded model output matches original model output.")
    
    import os
    os.remove(model_path)
    print(f"Cleaned up {model_path}.")
    
    print("\n--- Smoke Test Passed ---")