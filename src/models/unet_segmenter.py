# FILE: src/models/unet_segmenter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Type

class DoubleConv(nn.Module):
    """A block of two convolutional layers with batch normalization and ReLU activation."""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        """
        Initializes the DoubleConv block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (Optional[int], optional): Number of intermediate channels. 
                                                    Defaults to `out_channels`.
        """
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
        """Forward pass through the DoubleConv block."""
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling block with max pooling followed by a DoubleConv block."""
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the Downscaling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Downscaling block."""
        return self.maxpool_conv(x)

class AttentionGate(nn.Module):
    """Attention Gate to focus on relevant features from skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Initializes the Attention Gate.

        Args:
            F_g (int): Number of channels in the gating signal tensor.
            F_l (int): Number of channels in the skip connection tensor.
            F_int (int): Number of intermediate channels.
        """
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
        """
        Forward pass through the Attention Gate.

        Args:
            g (torch.Tensor): Gating signal from the decoder path.
            x (torch.Tensor): Skip connection tensor from the encoder path.

        Returns:
            torch.Tensor: The attended skip connection tensor.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    """Upscaling block followed by an optional attention gate and a DoubleConv block."""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, use_attention: bool = True):
        """
        Initializes the Upscaling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bilinear (bool, optional): If True, use bilinear upsampling, otherwise use transposed convolution. Defaults to True.
            use_attention (bool, optional): If True, apply an attention gate to the skip connection. Defaults to True.
        """
        super().__init__()
        self.use_attention = use_attention
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        if self.use_attention:
            self.att = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Upscaling block.

        Args:
            x1 (torch.Tensor): Tensor from the previous decoder layer.
            x2 (torch.Tensor): Skip connection tensor from the corresponding encoder layer.

        Returns:
            torch.Tensor: The output tensor after upsampling and convolution.
        """
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if self.use_attention:
            x2 = self.att(g=x1, x=x2)
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to map feature channels to the number of classes."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet model for semantic segmentation of geospatial imagery.

    This implementation includes an encoder-decoder architecture with skip connections.
    It supports both bilinear upsampling and transposed convolutions in the decoder,
    and can optionally use attention gates on the skip connections to improve focus
    on relevant features.
    """
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, use_attention: bool = True, bilinear: bool = True):
        """
        Initializes the UNet model.

        Args:
            in_channels (int): Number of input channels (e.g., 15 for Manthan).
            num_classes (int): Number of output classes for segmentation.
            base_channels (int, optional): Number of channels in the first convolutional layer. Defaults to 32.
            use_attention (bool, optional): If True, enables attention gates on skip connections. Defaults to True.
            bilinear (bool, optional): If True, uses bilinear upsampling in the decoder. Otherwise, uses transposed convolutions. Defaults to True.
        """
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
        """Initializes model weights using Kaiming normal distribution."""
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
        """Saves the model state and initialization arguments to a file."""
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
    def load(cls: Type['UNet'], path: str, device: Union[str, torch.device] = "cpu", **init_kwargs) -> 'UNet':
        """Loads a UNet model from a file."""
        checkpoint = torch.load(path, map_location=device)
        # Allow overriding saved kwargs
        model_kwargs = checkpoint.get('init_kwargs', {})
        model_kwargs.update(init_kwargs)
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)

def build_unet_segmenter(
    in_channels: int,
    num_classes: int,
    base_channels: int = 32,
    use_attention: bool = True,
    bilinear: bool = True,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None
) -> UNet:
    """
    Factory function to build and initialize the UNet model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        base_channels (int, optional): Number of base channels. Defaults to 32.
        use_attention (bool, optional): Whether to use attention gates. Defaults to True.
        bilinear (bool, optional): Whether to use bilinear upsampling. Defaults to True.
        device (Union[str, torch.device], optional): Device to move the model to. Defaults to "cpu".
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.

    Returns:
        UNet: The initialized UNet model.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        use_attention=use_attention,
        bilinear=bilinear
    )
    return model.to(device)


if __name__ == '__main__':
    print("--- Running UNet Segmenter Smoke Test ---")
    
    # --- Configuration ---
    B, C_IN, H, W = 1, 15, 256, 256
    NUM_CLASSES = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    print(f"Input tensor shape: ({B}, {C_IN}, {H}, {W})")
    print(f"Number of classes: {NUM_CLASSES}")

    # --- Model Creation ---
    print("\n1. Building UNet segmenter with Attention...")
    model = build_unet_segmenter(
        in_channels=C_IN,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        seed=42
    )

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
    model_path = "unet_segmenter_test.pth"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    loaded_model = UNet.load(model_path, device=DEVICE, in_channels=C_IN, num_classes=NUM_CLASSES)
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
