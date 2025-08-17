# FILE: src/models/species_recommender_model.py
import torch
import torch.nn as nn

class SpeciesNCF(nn.Module):
    """
    Neural Collaborative Filtering for Species-Site Matching.
    """
    def __init__(self, num_site_features: int, num_species: int, embedding_dim: int = 128):
        super().__init__()
        
        # Embedding layer for all species
        self.species_embedding = nn.Embedding(num_species, embedding_dim)
        
        # A network to process the site's geospatial features
        self.site_encoder = nn.Sequential(
            nn.Linear(num_site_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
        # The main interaction network
        self.interaction_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1) # Output a single suitability score (logit)
        )

    def forward(self, site_features: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates the suitability score for site-species pairs.
        
        Args:
            site_features: Tensor of shape [batch_size, num_site_features]
            species_ids: Tensor of shape [batch_size]
        """
        # Get embeddings for the site and the species
        site_vec = self.site_encoder(site_features)
        species_vec = self.species_embedding(species_ids)
        
        # Concatenate them to represent the interaction
        combined_vec = torch.cat([site_vec, species_vec], dim=1)
        
        # Get the final suitability score
        suitability_logit = self.interaction_mlp(combined_vec)
        
        return suitability_logit