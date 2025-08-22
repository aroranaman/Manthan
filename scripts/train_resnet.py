import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from collections import Counter
from manthan_core.models.contextual_resnet import LandUseClassifier
from manthan_core.data.image_dataset import ManthanDataset, get_transforms

def create_label_mappings(image_dir: str) -> dict:
    """Scans the dataset directory to create mappings for land use."""
    landuse_names = set()
    
    for filename in os.listdir(image_dir):
        if filename.endswith(('.tif', '.tiff', '.png')):
            lulc_match = re.search(r"_lulc_([a-zA-Z_]+)\.", filename)
            if lulc_match:
                landuse_names.add(lulc_match.group(1))

    landuse_map = {name: i for i, name in enumerate(sorted(list(landuse_names)))}
    
    with open('assets/landuse_map.json', 'w') as f:
        json.dump(landuse_map, f, indent=2)
        
    return landuse_map

def collate_fn_skip_corrupt(batch):
    """Filters out None values from a batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    """Main function to orchestrate the model training process."""
    print("--- 🚀 Starting Focused Land-Use Classifier Training ---")

    # --- 1. Configuration ---
    DATA_DIR = 'assets/Manthan_Training_Data'
    LEARNING_RATE = 3e-4 # Adjusted learning rate
    BATCH_SIZE = 32
    EPOCHS = 50 # Increased epochs
    VAL_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Training data not found in '{DATA_DIR}'.")
        return

    # --- 2. Create Label Mappings and Analyze Imbalance ---
    print("Scanning data and creating label mappings...")
    landuse_map = create_label_mappings(DATA_DIR)
    NUM_LANDUSE_CLASSES = len(landuse_map)
    
    # Analyze class imbalance
    all_labels = [re.search(r"_lulc_([a-zA-Z_]+)\.", f).group(1) for f in os.listdir(DATA_DIR) if re.search(r"_lulc_([a-zA-Z_]+)\.", f)]
    class_counts = Counter(all_labels)
    print(f"Found {NUM_LANDUSE_CLASSES} unique land use classes. Distribution: {class_counts}")
    
    # Calculate class weights to handle imbalance
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / class_counts[name] for name in sorted(landuse_map.keys())], dtype=torch.float).to(DEVICE)
    print(f"Calculated class weights: {class_weights}")

    # --- 3. Load Data ---
    print("Loading and splitting the dataset...")
    train_transforms = get_transforms(is_train=True)
    val_transforms = get_transforms(is_train=False)
    
    full_dataset = ManthanDataset(image_dir=DATA_DIR, landuse_map=landuse_map, transform=train_transforms)
    
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transforms to the validation set
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn_skip_corrupt)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn_skip_corrupt)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 4. Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    model = LandUseClassifier(num_landuse_classes=NUM_LANDUSE_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights) # Use class weights
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Use AdamW with weight decay

    # --- 5. Training Loop ---
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, landuse_labels in train_loader:
            if images is None: continue
            images, landuse_labels = images.to(DEVICE), landuse_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, landuse_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {running_loss/len(train_loader):.4f}")

        # --- Validation Loop ---
        model.eval()
        correct_landuse, total = 0, 0
        with torch.no_grad():
            for images, landuse_labels in val_loader:
                if images is None: continue
                images, landuse_labels = images.to(DEVICE), landuse_labels.to(DEVICE)
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += landuse_labels.size(0)
                correct_landuse += (predicted == landuse_labels).sum().item()
        
        if total > 0:
            print(f"  Validation Accuracy - Land Use: {100 * correct_landuse / total:.2f}%")

    # --- 6. Save Model ---
    print("Saving trained model...")
    torch.save(model.state_dict(), 'assets/landuse_classifier.pth')
    print("✅ Training complete. Model saved to 'assets/landuse_classifier.pth'.")

if __name__ == "__main__":
    main()