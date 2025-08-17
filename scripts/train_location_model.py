# FILE: scripts/train_location_model.py
import sys, re, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# --- Robust Path Fix ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Fix ---

from src.models.location_context_model import LocationLandUseAI
from src.data.geo_dataset import ManthanGeoDataset

# === CONFIG ===
DATA_DIR = project_root / "src" / "data"
IMAGE_DIR = DATA_DIR / "training_patches"
SAVED_MODEL_DIR = project_root / "saved_models"
SAVED_MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
INPUT_BANDS = 4          # B2,B3,B4,B8
NUM_LAND_USE_CLASSES = 3 # 0=Other/Urban, 1=Rural, 2=Forested

MANIFEST_PATH = DATA_DIR / "training_manifest.csv"
MAPPING_PATH = DATA_DIR / "district_mapping.json"

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


# ---------- helpers ----------
def _scan_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(files)

def _parse_ids_from_name(name: str) -> tuple[int | None, int | None]:
    d = re.search(r"(?:^|[_-])d(\d+)(?:[_-]|$)", name, re.IGNORECASE)
    lu = re.search(r"(?:^|[_-])lu(\d+)(?:[_-]|$)", name, re.IGNORECASE)
    did = int(d.group(1)) if d else None
    luid = int(lu.group(1)) if lu else None
    return did, luid

def _load_num_districts(mapping_path: Path) -> int:
    if not mapping_path.exists():
        # create a minimal mapping if missing (lets you train the loop)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump({"DummyDistrict": 0}, f, indent=2)
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return max(1, len(mapping))

def _synthesize_png_rgba(path: Path, h=64, w=64, seed=None):
    """Create a synthetic 4-band (RGBA) image scaled 0-255."""
    rng = np.random.default_rng(seed)
    arr = rng.random((h, w, 4), dtype=np.float32)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    img.save(path)

def _maybe_make_synthetic_dataset(image_dir: Path, count: int = 60, num_districts: int = 10):
    """If folder has no images, create a small labeled synthetic set."""
    files = _scan_images(image_dir)
    if files:
        return

    print(f"🧪 No training images found in {image_dir}. Creating {count} synthetic 4-band PNG patches…")
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        did = i % num_districts
        luid = i % NUM_LAND_USE_CLASSES
        out = image_dir / f"patch_{i:04d}_d{did}_lu{luid}.png"
        _synthesize_png_rgba(out, h=64, w=64, seed=42 + i)

def _build_or_fix_manifest(manifest_path: Path, image_dir: Path, num_districts: int) -> pd.DataFrame:
    needs_rebuild = True
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            if set(["image_filename", "district_id", "land_use_id"]).issubset(df.columns) and len(df) > 0:
                return df
        except Exception:
            needs_rebuild = True

    print("⚠️  Manifest missing/empty/invalid. Auto-building from image folder…")
    files = _scan_images(image_dir)
    if not files:
        raise RuntimeError(
            f"❌ No images found in:\n  {image_dir}\nSupported: {sorted(IMAGE_EXTS)}"
        )

    rows = []
    for p in files:
        did, luid = _parse_ids_from_name(p.name)
        if did is None:
            did = 0
        if luid is None:
            luid = random.choice(range(NUM_LAND_USE_CLASSES))
        did = max(0, min(did, max(0, num_districts - 1)))
        luid = max(0, min(luid, NUM_LAND_USE_CLASSES - 1))
        rows.append({"image_filename": p.name, "district_id": did, "land_use_id": luid})

    df = pd.DataFrame(rows, columns=["image_filename", "district_id", "land_use_id"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"✅ Wrote new manifest with {len(df)} rows → {manifest_path}")
    return df


# ---------- training ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_districts = _load_num_districts(MAPPING_PATH)

    # If empty, create a tiny synthetic dataset so the loop runs.
    _maybe_make_synthetic_dataset(IMAGE_DIR, count=60, num_districts=num_districts)

    _ = _build_or_fix_manifest(MANIFEST_PATH, IMAGE_DIR, num_districts)

    # Create dataset (expects manifest + images)
    full_dataset = ManthanGeoDataset(
        manifest_path=MANIFEST_PATH,
        image_dir=IMAGE_DIR,
        transform=None  # use your dataset's internal transforms if any
    )
    if len(full_dataset) == 0:
        raise RuntimeError("❌ The dataset is empty. Check paths and image formats.")

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = LocationLandUseAI(
        num_input_channels=INPUT_BANDS,
        num_districts=num_districts,
        num_land_use_classes=NUM_LAND_USE_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def _run_epoch(loader, train_mode: bool):
        model.train(mode=train_mode)
        total, batches = 0.0, 0
        for images, district_labels, land_use_labels in loader:
            if not torch.is_floating_point(images):
                images = images.float()
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim != 4:
                raise RuntimeError(f"Bad image batch shape: {images.shape} (expected [B,C,H,W])")
            if images.shape[1] != INPUT_BANDS:
                raise RuntimeError(
                    f"Expected {INPUT_BANDS} channels but got {images.shape[1]}. "
                    "Verify dataset reads 4 channels (PNG saved as RGBA or 4-band TIFF)."
                )

            images = images.to(device, non_blocking=True)
            district_labels = district_labels.to(device, non_blocking=True).long()
            land_use_labels = land_use_labels.to(device, non_blocking=True).long()

            if train_mode:
                optimizer.zero_grad()

            loc_logits, lu_logits = model(images)
            loss = criterion(loc_logits, district_labels) + 0.5 * criterion(lu_logits, land_use_labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total += loss.item()
            batches += 1
        return total / max(1, batches)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = _run_epoch(train_loader, True)
        va_loss = _run_epoch(val_loader, False)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            out_path = SAVED_MODEL_DIR / "location_context_model.pth"
            torch.save(model.state_dict(), out_path)
            print(f"💾 Saved best model (val_loss={best_val:.4f}) → {out_path}")

    print("✅ Training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise
