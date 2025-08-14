# FILE: src/models/ncf_species.py
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Type

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model for predicting species suitability.

    This model combines embeddings for regions and species with dense environmental
    features to produce a suitability score. It is designed to learn the latent
    interactions between geographical regions, species characteristics, and local
    environmental conditions.
    """
    def __init__(
        self,
        num_regions: int,
        num_species: int,
        env_dim: int,
        emb_dim: int = 16,
        hidden: int = 64,
        dropout: float = 0.1
    ):
        """
        Initializes the NCF model.

        Args:
            num_regions (int): The total number of unique regions (e.g., ecoregions or states).
            num_species (int): The total number of unique species in the database.
            env_dim (int): The dimensionality of the environmental feature vector (e.g., rainfall, temp, soil pH).
            emb_dim (int, optional): The dimensionality of the region and species embeddings. Defaults to 16.
            hidden (int, optional): The number of units in the hidden MLP layer. Defaults to 64.
            dropout (float, optional): The dropout rate to apply. Defaults to 0.1.
        """
        super().__init__()
        self.num_regions = num_regions
        self.num_species = num_species
        self.env_dim = env_dim
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout_rate = dropout

        # Embedding layers for categorical features
        self.region_embedding = nn.Embedding(num_regions, emb_dim)
        self.species_embedding = nn.Embedding(num_species, emb_dim)

        # MLP layers to process the combined features
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim + env_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, region_ids: torch.LongTensor, species_ids: torch.LongTensor, env_feats: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass for a batch of region-species-environment interactions.

        Args:
            region_ids (torch.LongTensor): A tensor of region IDs. Shape: (B,).
            species_ids (torch.LongTensor): A tensor of species IDs. Shape: (B,).
            env_feats (torch.FloatTensor): A tensor of environmental features. Shape: (B, env_dim).

        Returns:
            torch.FloatTensor: A tensor of suitability scores in the range [0, 1]. Shape: (B,).
        """
        region_emb = self.region_embedding(region_ids)
        species_emb = self.species_embedding(species_ids)
        
        # Concatenate embeddings and environmental features
        x = torch.cat([region_emb, species_emb, env_feats], dim=-1)
        
        # Pass through MLP
        logits = self.mlp(x)
        
        # Apply sigmoid to get the final suitability score
        scores = self.sigmoid(logits).squeeze(-1)
        return scores

    def predict_topk(self, region_id: int, env_feats: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the top-k most suitable species for a given region and environment.

        This method efficiently calculates suitability scores for all species for the
        given region and returns the ones with the highest scores.

        Args:
            region_id (int): The ID of the region to predict for.
            env_feats (torch.Tensor): The environmental feature tensor for the location. Shape: (env_dim,).
            k (int, optional): The number of top species to return. Defaults to 10.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - topk_indices (torch.Tensor): The indices (IDs) of the top-k species. Shape: (k,).
                - topk_scores (torch.Tensor): The suitability scores of the top-k species. Shape: (k,).
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Prepare inputs for all species in a batch
            all_species_ids = torch.arange(self.num_species, device=device)
            region_ids = torch.tensor([region_id] * self.num_species, device=device)
            env_feats_batch = env_feats.unsqueeze(0).repeat(self.num_species, 1).to(device)
            
            # Get scores for all species
            all_scores = self.forward(region_ids, all_species_ids, env_feats_batch)
            
            # Get top-k scores and their corresponding indices
            topk_scores, topk_indices = torch.topk(all_scores, k=min(k, self.num_species))
            
        return topk_indices, topk_scores

    def save(self, path: str) -> None:
        """Saves the model state and initialization arguments to a file."""
        torch.save({
            'init_kwargs': {
                'num_regions': self.num_regions,
                'num_species': self.num_species,
                'env_dim': self.env_dim,
                'emb_dim': self.emb_dim,
                'hidden': self.hidden,
                'dropout': self.dropout_rate
            },
            'model_state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls: Type['NCFModel'], path: str, **init_kwargs) -> 'NCFModel':
        """Loads an NCFModel from a file."""
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        # Use provided kwargs to override saved kwargs if necessary
        model_kwargs = checkpoint.get('init_kwargs', {})
        model_kwargs.update(init_kwargs)
        
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

def build_ncf_species_model(
    num_regions: int,
    num_species: int,
    env_dim: int,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    **kwargs
) -> NCFModel:
    """
    Factory function to build and initialize the NCFModel for species suitability.

    Args:
        num_regions (int): Total number of unique regions.
        num_species (int): Total number of unique species.
        env_dim (int): Dimensionality of the environmental feature vector.
        device (Union[str, torch.device], optional): Device to move the model to. Defaults to "cpu".
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        **kwargs: Additional arguments to pass to the NCFModel constructor.

    Returns:
        NCFModel: The initialized NCF model.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = NCFModel(num_regions, num_species, env_dim, **kwargs)
    return model.to(device)


if __name__ == '__main__':
    print("--- Running NCF Species Model Smoke Test ---")
    
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
    model = build_ncf_species_model(
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
    model_path = "ncf_species_test.pth"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    loaded_model = NCFModel.load(
        model_path,
        num_regions=NUM_REGIONS,
        num_species=NUM_SPECIES,
        env_dim=ENV_DIM
    )
    loaded_model.to(DEVICE)
    print("Model loaded successfully.")
    
    loaded_scores = loaded_model(region_ids, species_ids, env_feats)
    assert torch.allclose(scores, loaded_scores, atol=1e-6), "Mismatch between original and loaded model outputs."
    print("✅ Loaded model output matches original model output.")
    
    import os
    os.remove(model_path)
    print(f"Cleaned up {model_path}.")
    
    print("\n--- Smoke Test Passed ---")