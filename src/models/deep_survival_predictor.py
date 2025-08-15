# FILE: src/models/deep_survival_predictor.py
import torch
import torch.nn as nn
from typing import Optional, Union, Type

class DeepSurvivalPredictor(nn.Module):
    """
    A deep learning model to predict the survival probability of a species
    in a given environment, as specified in the Manthan AI Framework.
    """
    def __init__(
        self,
        num_species: int,
        env_features_dim: int = 47,
        species_embedding_dim: int = 128,
        hidden_dims: list[int] = [512, 256, 128],
        dropout_rate: float = 0.3
    ):
        """
        Initializes the Deep Survival Predictor model.

        Args:
            num_species (int): The total number of unique species in the database.
            env_features_dim (int): The number of environmental & socioeconomic features.
            species_embedding_dim (int): The dimensionality of the species embedding.
            hidden_dims (list[int]): A list of hidden layer sizes for the dense network.
            dropout_rate (float): The dropout rate to apply between layers.
        """
        super().__init__()
        self.num_species = num_species
        self.env_features_dim = env_features_dim
        self.species_embedding_dim = species_embedding_dim

        # Embedding layer for species characteristics
        self.species_embedding = nn.Embedding(num_species, species_embedding_dim)

        # Create the main dense network
        layers = []
        input_dim = env_features_dim + species_embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        self.dense_layers = nn.Sequential(*layers)

        # Output layer for survival probability (a single value between 0 and 1)
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, species_ids: torch.LongTensor, env_features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the model.

        Args:
            species_ids (torch.LongTensor): A tensor of species IDs. Shape: (B,).
            env_features (torch.FloatTensor): A tensor of environmental features. Shape: (B, env_features_dim).

        Returns:
            torch.FloatTensor: A tensor of predicted survival probabilities. Shape: (B, 1).
        """
        # Get species embeddings
        species_embedded = self.species_embedding(species_ids)
        
        # Concatenate environmental features with species embeddings
        combined_features = torch.cat([species_embedded, env_features], dim=1)
        
        # Pass through dense layers
        hidden_output = self.dense_layers(combined_features)
        
        # Get final prediction
        logits = self.output_layer(hidden_output)
        survival_probability = self.sigmoid(logits)
        
        return survival_probability

def build_deep_survival_model(
    num_species: int,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    **kwargs
) -> DeepSurvivalPredictor:
    """Factory function to build and initialize the DeepSurvivalPredictor."""
    if seed is not None:
        torch.manual_seed(seed)
    model = DeepSurvivalPredictor(num_species=num_species, **kwargs)
    return model.to(device)

if __name__ == '__main__':
    # --- Smoke Test for the Deep Survival Predictor ---
    print("--- Initializing Deep Survival Predictor ---")
    
    # Configuration
    NUM_SPECIES = 15000
    ENV_FEATURES = 47
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build model
    model = build_deep_survival_model(num_species=NUM_SPECIES, device=DEVICE, seed=42)
    print(f"✅ Model initialized successfully on {DEVICE}.")
    
    # Create synthetic input data
    species_ids = torch.randint(0, NUM_SPECIES, (BATCH_SIZE,), device=DEVICE)
    env_features = torch.randn(BATCH_SIZE, ENV_FEATURES, device=DEVICE)
    print(f"🧠 Created synthetic input data with shapes: species_ids={species_ids.shape}, env_features={env_features.shape}")
    
    # Perform inference
    with torch.no_grad():
        model.eval()
        predictions = model(species_ids, env_features)
    
    print("🚀 Inference complete.")
    print(f"   - Output predictions shape: {predictions.shape}")
    
    expected_shape = (BATCH_SIZE, 1)
    assert predictions.shape == expected_shape, "Output shape is incorrect!"
    assert predictions.min() >= 0 and predictions.max() <= 1, "Output probabilities are not in [0, 1] range!"
    print("✅ Output shape and value range are correct.")
    print("\n--- Smoke Test Passed ---")
