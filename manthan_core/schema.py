# /schemas.py
# Unified and consolidated schema for the Manthan system.

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any

# --- Core Statistical and Geospatial Models ---

class Stats(BaseModel):
    """A simple model for statistical summaries, as provided by the user."""
    mean: float
    median: float
    std_dev: float

class AOI(BaseModel):
    """
    Defines the Area of Interest (AOI) for analysis.
    Based on the user's AOI.py.
    """
    name: str = Field(..., description="A unique name or ID for the AOI.")
    geojson: Dict[str, Any] = Field(..., description="A GeoJSON dictionary representing the AOI polygon.")

# --- Site and Species Models ---

class SiteFingerprint(BaseModel):
    """
    The authoritative data contract for a site's ecological profile.
    Combines the user's site_fingerprint.py with more detailed fields.
    """
    aoi: AOI = Field(..., description="The Area of Interest this fingerprint corresponds to.")
    
    # Detailed Climate Data
    avg_annual_rainfall_mm: float = Field(..., description="Average annual rainfall in mm.")
    avg_annual_temp_c: float = Field(..., description="Average annual temperature in Celsius.")
    climatic_water_balance: float = Field(..., description="Precipitation minus potential evapotranspiration.")

    # Detailed Soil Data
    avg_soil_ph: float = Field(..., description="Average topsoil pH.")
    avg_soil_organic_carbon_pct: float = Field(..., description="Average soil organic carbon in percent.")
    dominant_soil_texture: str = Field(..., description="Dominant soil texture (e.g., 'loamy').")

    # Topography
    avg_elevation_m: float = Field(..., description="Average elevation in meters.")
    slope_stats: Stats = Field(..., description="Topographic slope statistics in degrees.")
    
    # Vegetation Indices with seasonal stats
    ndvi_stats: Dict[str, Stats] = Field(..., description="Seasonal NDVI statistics (e.g., {'pre_monsoon': Stats, ...}).")
    evi_stats: Optional[Dict[str, Stats]] = Field(None, description="Optional seasonal EVI statistics.")

class Species(BaseModel):
    """
    Represents a single plant species with its full ecological and economic attributes.
    """
    id: str = Field(..., description="Unique identifier for the species (e.g., 'acacia_nilotica').")
    scientific_name: str = Field(..., description="Scientific name (e.g., 'Acacia nilotica').")
    common_name: str = Field(..., description="Common name (e.g., 'Babul').")
    miyawaki_layer: str = Field(..., description="Miyawaki forest layer (Canopy, Sub-Canopy, Shrub, Ground Cover).")
    
    # Climate Envelope Tolerances
    min_rainfall_mm: float = Field(..., description="Minimum annual rainfall tolerance in mm.")
    max_rainfall_mm: float = Field(..., description="Maximum annual rainfall tolerance in mm.")
    min_temp_c: float = Field(..., description="Minimum average temperature tolerance in Celsius.")
    max_temp_c: float = Field(..., description="Maximum average temperature tolerance in Celsius.")
    
    # Soil Tolerances
    min_ph: float = Field(..., description="Minimum soil pH tolerance.")
    max_ph: float = Field(..., description="Maximum soil pH tolerance.")
    compatible_soil_textures: List[str] = Field(..., description="List of compatible soil textures.")

    # Economic & Ecological Attributes
    is_native: bool = Field(True, description="Whether the species is native to the region.")
    is_invasive: bool = Field(False, description="Whether the species is known to be invasive.")
    is_nitrogen_fixer: bool = Field(False, description="Whether the species is a nitrogen-fixer.")
    description: str = Field("", description="A brief description of the species.")
    
    # Interspecies Compatibility
    compatibility_scores: Dict[str, float] = Field({}, description="Scores indicating interaction with other species.")

# --- Recommendation and Planning Models ---

class SpeciesRecommendation(BaseModel):
    """
    Data contract for a single species recommendation, linking a Species to a Site.
    """
    species: Species
    score: float = Field(..., ge=0, le=1, description="Overall compatibility score from 0 to 1.")
    why: str = Field(..., description="Ecological and economic reasoning for the recommendation.")
    scores_breakdown: Dict[str, float] = Field(..., description="Breakdown of scores (water, soil, climate).")

class RestorationPlan(BaseModel):
    """
    A complete restoration plan for an AOI, including recommendations and forecasts.
    """
    site_fingerprint: SiteFingerprint
    species_recommendations: List[SpeciesRecommendation]
    
    # Miyawaki Forest Composition
    suggested_composition: Dict[str, List[str]] = Field(..., description="Suggested species scientific names for each Miyawaki layer.")
    planting_density_per_sqm: float = Field(..., description="Recommended planting density per square meter.")
    
    # Detailed Forecasts
    predicted_survival_rate_pct: float = Field(..., description="Predicted 20-year survival probability.")
    survival_prediction_interval: Tuple[float, float] = Field(..., description="Prediction interval (e.g., 10th, 90th percentile) for survival.")
    time_to_canopy_closure_years: int = Field(..., description="Predicted time to canopy closure in years.")
    predicted_biomass_ton_per_ha: float = Field(..., description="Predicted biomass accumulation in tons per hectare after 20 years.")
    predicted_carbon_seq_ton_per_ha: float = Field(..., description="Predicted carbon sequestration in tons per hectare after 20 years.")
    cost_benefit_analysis: Dict[str, float] = Field(..., description="Summary of cost-benefit analysis (e.g., NPV).")

if __name__ == '__main__':
    # Example of how to create an instance of the final, unified RestorationPlan
    
    # 1. Define Stats and AOI
    slope_stats = Stats(mean=10.5, median=9.8, std_dev=2.1)
    ndvi_stats_mock = {"pre_monsoon": Stats(mean=0.3, median=0.28, std_dev=0.05)}

    # Use a valid polygon (simple 0.01° square near Delhi) instead of ellipsis
    aoi_polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [77.0, 28.0],
                [77.01, 28.0],
                [77.01, 28.01],
                [77.0, 28.01],
                [77.0, 28.0]
            ]
        ]
    }
    aoi_mock = AOI(name="Test_Site_01", geojson=aoi_polygon)

    # 2. Define SiteFingerprint
    site_fp = SiteFingerprint(
        aoi=aoi_mock,
        avg_annual_rainfall_mm=1200,
        avg_annual_temp_c=25,
        climatic_water_balance=300,
        avg_soil_ph=6.8,
        avg_soil_organic_carbon_pct=1.2,
        dominant_soil_texture='loamy',
        avg_elevation_m=500,
        slope_stats=slope_stats,
        ndvi_stats=ndvi_stats_mock
    )

    # 3. Define a Species
    khair_species = Species(
        id='acacia_catechu',
        scientific_name='Acacia catechu',
        common_name='Khair',
        miyawaki_layer='Canopy',
        min_rainfall_mm=500,
        max_rainfall_mm=2000,
        min_temp_c=15,
        max_temp_c=40,
        min_ph=5.5,
        max_ph=8.0,
        compatible_soil_textures=['sandy', 'loamy'],
        is_nitrogen_fixer=True
    )

    # 4. Define a Recommendation
    rec = SpeciesRecommendation(
        species=khair_species,
        score=0.85,
        why="Excellent match for site's rainfall and temperature. As a nitrogen-fixer, it will improve soil fertility.",
        scores_breakdown={'water': 0.9, 'soil': 0.8, 'climate': 0.85}
    )

    # 5. Define the full RestorationPlan
    final_plan = RestorationPlan(
        site_fingerprint=site_fp,
        species_recommendations=[rec],
        suggested_composition={'Canopy': ['Acacia catechu'], 'Shrub': []},
        planting_density_per_sqm=3.0,
        predicted_survival_rate_pct=85.5,
        survival_prediction_interval=(78.0, 91.0),
        time_to_canopy_closure_years=7,
        predicted_biomass_ton_per_ha=150.0,
        predicted_carbon_seq_ton_per_ha=75.0,
        cost_benefit_analysis={'net_present_value': 50000.0}
    )
    
    print("--- Unified Restoration Plan Example ---")
    print(final_plan.model_dump_json(indent=2))
