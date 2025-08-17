# FILE: src/core/knowledge_graph.py

import pandas as pd
from pathlib import Path

class KnowledgeGraph:
    """
    A data hub that loads and provides query access to aggregated data from
    authoritative Indian biodiversity sources like IBP and BSI.
    """
    def __init__(self, data_dir: Path):
        print("Initializing Knowledge Graph...")
        self.data_dir = data_dir
        self._load_data()
        print(f"✅ Knowledge Graph loaded with {len(self.master_df)} species records.")

    def _load_data(self):
        """Loads and merges data from multiple source files."""
        # Load the base checklist (e.g., from IBP)
        checklist_path = self.data_dir / "ibp_uttarakhand_checklist.csv"
        if not checklist_path.exists():
            raise FileNotFoundError(f"Checklist file not found at {checklist_path}")
        
        checklist_df = pd.read_csv(checklist_path)

        # Load the enrichment data (e.g., from BSI/eFlora)
        status_path = self.data_dir / "bsi_conservation_status.csv"
        if not status_path.exists():
            raise FileNotFoundError(f"Conservation status file not found at {status_path}")
            
        status_df = pd.read_csv(status_path)

        # Merge the dataframes to create a single, unified knowledge base
        self.master_df = pd.merge(checklist_df, status_df, on="scientific_name", how="left")
        
        # Fill missing values with safe defaults
        self.master_df['conservation_status'].fillna('Not Evaluated', inplace=True)
        self.master_df['is_invasive'].fillna(False, inplace=True)
        self.master_df['is_threatened'].fillna(False, inplace=True)

    def get_all_species(self) -> pd.DataFrame:
        """Returns the complete list of all species in the knowledge base."""
        return self.master_df

    def apply_ecological_rules(self, site_features: dict) -> pd.DataFrame:
        """
        Filters the master species list based on a set of ecological rules.
        """
        candidates = self.master_df.copy()
        print("\n--- Applying Ecological Rules ---")

        # Rule 1: Filter out known invasive species
        initial_count = len(candidates)
        candidates = candidates[candidates['is_invasive'] == False]
        print(f"Rule 1 (Invasives): Removed {initial_count - len(candidates)} species. Candidates remaining: {len(candidates)}")

        # Rule 2: Filter by elevation
        elevation = site_features.get('elevation_mean', 0)
        if elevation > 0:
            initial_count = len(candidates)
            # Drop species that have a max_elevation defined and where the site is too high
            candidates.dropna(subset=['max_elevation_m'], inplace=True)
            candidates = candidates[candidates['max_elevation_m'] >= elevation]
            
            # Drop species that have a min_elevation defined and where the site is too low
            candidates.dropna(subset=['min_elevation_m'], inplace=True)
            candidates = candidates[candidates['min_elevation_m'] <= elevation]
            print(f"Rule 2 (Elevation): Filtered for elevation ~{elevation:.0f}m. Candidates remaining: {len(candidates)}")
        
        # Add more rules here for rainfall, soil pH, etc.

        return candidates