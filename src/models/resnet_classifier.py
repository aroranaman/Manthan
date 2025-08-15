# FILE: src/models/resnet_classifier.py
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
        num_input_channels (int): Number of channels in the input tensor (e.g., 15 for Manthan).
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
                new_conv1.weight.data.fill_(0.0)
                new_conv1.weight.data[:, :3, :, :] = original_weights
                for i in range(3, num_input_channels):
                    new_conv1.weight.data[:, i, :, :] = weight_mean.squeeze(1)


        model.conv1 = new_conv1

    # Replace the final fully connected layer for the new classification task
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )

    return model.to(device)

if __name__ == '__main__':
    print("--- Running ResNet Classifier Smoke Test ---")
    
    # --- Configuration ---
    B, C, H, W = 2, 15, 224, 224
    NUM_CLASSES = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    print(f"Input tensor shape: ({B}, {C}, {H}, {W})")
    print(f"Number of classes: {NUM_CLASSES}")

    # --- Model Creation ---
    print("\n1. Building model with inflated pretrained weights...")
    model_pretrained = build_resnet_veg(
        num_input_channels=C,
        num_classes=NUM_CLASSES,
        pretrained=True,
        device=DEVICE,
        seed=42
    )

    # --- Forward Pass ---
    print("\n2. Performing forward pass...")
    synthetic_input = torch.randn(B, C, H, W, device=DEVICE)
    
    with torch.no_grad():
        model_pretrained.eval()
        logits = model_pretrained(synthetic_input)

    # --- Shape Verification ---
    print(f"Input shape: {synthetic_input.shape}")
    print(f"Output (logits) shape: {logits.shape}")
    
    expected_shape = (B, NUM_CLASSES)
    assert logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {logits.shape}"
    print("✅ Output shape is correct.")

    # --- Save/Load Test ---
    print("\n3. Testing model serialization...")
    model_path = "resnet_classifier_test.pth"
    save(model_pretrained, model_path)
    print(f"Model saved to {model_path}")
    
    loaded_model = load(model_path, num_input_channels=C, num_classes=NUM_CLASSES, device=DEVICE)
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
    print("The ResNet Classifier is functioning correctly.")