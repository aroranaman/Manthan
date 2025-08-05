"""
Indian Forest Species Database
Comprehensive database of native Indian tree species with ecological parameters
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Species:
    """Species data model with ecological parameters"""
    scientific_name: str
    common_name: str
    family: str
    native_regions: List[str]
    forest_types: List[str]
    canopy_layer: str  # emergent, canopy, understory, ground
    growth_rate: str  # slow, medium, fast
    rainfall_min: float  # mm/year
    rainfall_max: float
    temperature_min: float  # Celsius
    temperature_max: float
    soil_ph_min: float
    soil_ph_max: float
    elevation_min: float  # meters
    elevation_max: float
    carbon_sequestration: float  # kg CO2/year/tree
    ecological_value: List[str]  # wildlife, nitrogen-fixing, medicinal, etc.
    planting_notes: str

def get_species_database() -> Dict[str, Species]:
    """
    Returns comprehensive database of Indian forest species
    Organized by scientific name
    """
    
    species_db = {
        # WESTERN GHATS SPECIES
        "Cullenia exarillata": Species(
            scientific_name="Cullenia exarillata",
            common_name="Wild Durian",
            family="Malvaceae",
            native_regions=["Western Ghats"],
            forest_types=["Tropical Wet Evergreen"],
            canopy_layer="emergent",
            growth_rate="slow",
            rainfall_min=2000, rainfall_max=7000,
            temperature_min=18, temperature_max=32,
            soil_ph_min=4.5, soil_ph_max=6.5,
            elevation_min=600, elevation_max=1800,
            carbon_sequestration=350,
            ecological_value=["keystone", "wildlife", "endemic"],
            planting_notes="Requires high humidity and shade when young"
        ),
        
        "Myristica dactyloides": Species(
            scientific_name="Myristica dactyloides",
            common_name="Wild Nutmeg",
            family="Myristicaceae",
            native_regions=["Western Ghats"],
            forest_types=["Tropical Wet Evergreen"],
            canopy_layer="canopy",
            growth_rate="medium",
            rainfall_min=2500, rainfall_max=6000,
            temperature_min=20, temperature_max=30,
            soil_ph_min=5.0, soil_ph_max=6.5,
            elevation_min=200, elevation_max=1200,
            carbon_sequestration=280,
            ecological_value=["endemic", "threatened", "wildlife"],
            planting_notes="Swampy areas preferred, needs consistent moisture"
        ),
        
        "Holigarna arnottiana": Species(
            scientific_name="Holigarna arnottiana",
            common_name="Black Varnish Tree",
            family="Anacardiaceae",
            native_regions=["Western Ghats"],
            forest_types=["Tropical Wet Evergreen", "Tropical Semi-Evergreen"],
            canopy_layer="canopy",
            growth_rate="medium",
            rainfall_min=2000, rainfall_max=5000,
            temperature_min=20, temperature_max=32,
            soil_ph_min=5.5, soil_ph_max=7.0,
            elevation_min=100, elevation_max=1500,
            carbon_sequestration=260,
            ecological_value=["endemic", "timber", "wildlife"],
            planting_notes="Caution: sap causes severe allergic reactions"
        ),
        
        # CENTRAL INDIAN SPECIES
        "Shorea robusta": Species(
            scientific_name="Shorea robusta",
            common_name="Sal",
            family="Dipterocarpaceae",
            native_regions=["Central India", "Eastern India"],
            forest_types=["Tropical Moist Deciduous", "Tropical Dry Deciduous"],
            canopy_layer="emergent",
            growth_rate="medium",
            rainfall_min=1000, rainfall_max=3000,
            temperature_min=15, temperature_max=38,
            soil_ph_min=5.5, soil_ph_max=7.0,
            elevation_min=100, elevation_max=1500,
            carbon_sequestration=350,
            ecological_value=["dominant", "timber", "resin", "religious"],
            planting_notes="Forms pure stands, gregarious flowering every 3-5 years"
        ),
        
        "Tectona grandis": Species(
            scientific_name="Tectona grandis",
            common_name="Teak",
            family="Lamiaceae",
            native_regions=["Central India", "Western Ghats", "Eastern Ghats"],
            forest_types=["Tropical Dry Deciduous", "Tropical Moist Deciduous"],
            canopy_layer="canopy",
            growth_rate="medium",
            rainfall_min=800, rainfall_max=2500,
            temperature_min=20, temperature_max=40,
            soil_ph_min=6.0, soil_ph_max=7.5,
            elevation_min=0, elevation_max=1200,
            carbon_sequestration=250,
            ecological_value=["premium timber", "plantation"],
            planting_notes="Requires well-drained soils, fire resistant"
        ),
        
        "Madhuca longifolia": Species(
            scientific_name="Madhuca longifolia",
            common_name="Mahua",
            family="Sapotaceae",
            native_regions=["Central India", "Northern India"],
            forest_types=["Tropical Dry Deciduous"],
            canopy_layer="canopy",
            growth_rate="medium",
            rainfall_min=500, rainfall_max=1500,
            temperature_min=15, temperature_max=45,
            soil_ph_min=5.5, soil_ph_max=8.0,
            elevation_min=0, elevation_max=1200,
            carbon_sequestration=280,
            ecological_value=["edible flowers", "oil seeds", "cultural", "wildlife"],
            planting_notes="Drought tolerant, important for tribal communities"
        ),
        
        # RAJASTHAN/ARID ZONE SPECIES
        "Prosopis cineraria": Species(
            scientific_name="Prosopis cineraria",
            common_name="Khejri",
            family="Fabaceae",
            native_regions=["Rajasthan", "Gujarat", "Punjab", "Haryana"],
            forest_types=["Tropical Thorn", "Desert"],
            canopy_layer="canopy",
            growth_rate="slow",
            rainfall_min=100, rainfall_max=600,
            temperature_min=5, temperature_max=50,
            soil_ph_min=7.0, soil_ph_max=8.5,
            elevation_min=0, elevation_max=600,
            carbon_sequestration=150,
            ecological_value=["nitrogen fixing", "fodder", "sacred", "drought resistant"],
            planting_notes="State tree of Rajasthan, survives extreme drought"
        ),
        
        "Tecomella undulata": Species(
            scientific_name="Tecomella undulata",
            common_name="Rohida/Desert Teak",
            family="Bignoniaceae",
            native_regions=["Rajasthan", "Gujarat"],
            forest_types=["Desert", "Tropical Thorn"],
            canopy_layer="understory",
            growth_rate="slow",
            rainfall_min=150, rainfall_max=500,
            temperature_min=5, temperature_max=48,
            soil_ph_min=7.5, soil_ph_max=8.5,
            elevation_min=0, elevation_max=500,
            carbon_sequestration=120,
            ecological_value=["timber", "medicinal", "desert adapted"],
            planting_notes="Extremely drought tolerant, beautiful yellow flowers"
        ),
        
        # HIMALAYAN SPECIES
        "Quercus leucotrichophora": Species(
            scientific_name="Quercus leucotrichophora",
            common_name="Banj Oak",
            family="Fagaceae",
            native_regions=["Himalayas"],
            forest_types=["Montane Wet Temperate", "Subtropical Pine"],
            canopy_layer="canopy",
            growth_rate="slow",
            rainfall_min=1000, rainfall_max=3000,
            temperature_min=5, temperature_max=25,
            soil_ph_min=5.0, soil_ph_max=6.5,
            elevation_min=1200, elevation_max=2300,
            carbon_sequestration=320,
            ecological_value=["watershed protection", "fodder", "fuel"],
            planting_notes="Important for water conservation in hills"
        ),
        
        "Rhododendron arboreum": Species(
            scientific_name="Rhododendron arboreum",
            common_name="Burans",
            family="Ericaceae",
            native_regions=["Himalayas"],
            forest_types=["Montane Wet Temperate"],
            canopy_layer="understory",
            growth_rate="medium",
            rainfall_min=1500, rainfall_max=3500,
            temperature_min=0, temperature_max=20,
            soil_ph_min=4.5, soil_ph_max=6.0,
            elevation_min=1500, elevation_max=3600,
            carbon_sequestration=180,
            ecological_value=["ornamental", "medicinal", "honey"],
            planting_notes="State tree of Uttarakhand, prefers acidic soils"
        ),
        
        # EASTERN INDIA/NORTHEAST SPECIES
        "Dipterocarpus retusus": Species(
            scientific_name="Dipterocarpus retusus",
            common_name="Hollong",
            family="Dipterocarpaceae",
            native_regions=["Northeast India"],
            forest_types=["Tropical Wet Evergreen", "Tropical Semi-Evergreen"],
            canopy_layer="emergent",
            growth_rate="fast",
            rainfall_min=2000, rainfall_max=5000,
            temperature_min=15, temperature_max=35,
            soil_ph_min=5.0, soil_ph_max=6.5,
            elevation_min=100, elevation_max=1000,
            carbon_sequestration=400,
            ecological_value=["timber", "wildlife", "dominant"],
            planting_notes="State tree of Assam, fast growing"
        ),
        
        # MANGROVE SPECIES
        "Rhizophora mucronata": Species(
            scientific_name="Rhizophora mucronata",
            common_name="Asiatic Mangrove",
            family="Rhizophoraceae",
            native_regions=["Coastal India", "Sundarbans", "Andaman"],
            forest_types=["Mangrove"],
            canopy_layer="canopy",
            growth_rate="medium",
            rainfall_min=1000, rainfall_max=3000,
            temperature_min=20, temperature_max=35,
            soil_ph_min=6.0, soil_ph_max=8.5,
            elevation_min=-1, elevation_max=5,
            carbon_sequestration=280,
            ecological_value=["coastal protection", "fishery support", "carbon storage"],
            planting_notes="Requires tidal influence, prop roots"
        ),
        
        # COMMON PLANTATION SPECIES
        "Azadirachta indica": Species(
            scientific_name="Azadirachta indica",
            common_name="Neem",
            family="Meliaceae",
            native_regions=["All India"],
            forest_types=["Tropical Dry Deciduous", "Urban"],
            canopy_layer="canopy",
            growth_rate="fast",
            rainfall_min=400, rainfall_max=1200,
            temperature_min=20, temperature_max=45,
            soil_ph_min=6.0, soil_ph_max=8.5,
            elevation_min=0, elevation_max=1500,
            carbon_sequestration=180,
            ecological_value=["medicinal", "pesticide", "hardy"],
            planting_notes="Extremely adaptable, good for degraded lands"
        ),
        
        "Ficus benghalensis": Species(
            scientific_name="Ficus benghalensis",
            common_name="Banyan",
            family="Moraceae",
            native_regions=["All India"],
            forest_types=["All except Alpine"],
            canopy_layer="emergent",
            growth_rate="medium",
            rainfall_min=500, rainfall_max=2500,
            temperature_min=10, temperature_max=45,
            soil_ph_min=5.5, soil_ph_max=8.0,
            elevation_min=0, elevation_max=1800,
            carbon_sequestration=400,
            ecological_value=["keystone", "wildlife", "cultural", "shade"],
            planting_notes="National tree of India, aerial roots need space"
        ),
        
        "Terminalia arjuna": Species(
            scientific_name="Terminalia arjuna",
            common_name="Arjun",
            family="Combretaceae",
            native_regions=["Central India", "Northern India"],
            forest_types=["Riparian", "Tropical Moist Deciduous"],
            canopy_layer="canopy",
            growth_rate="medium",
            rainfall_min=750, rainfall_max=2000,
            temperature_min=15, temperature_max=40,
            soil_ph_min=6.0, soil_ph_max=8.0,
            elevation_min=0, elevation_max=1200,
            carbon_sequestration=300,
            ecological_value=["medicinal", "riverbank protection", "sericulture"],
            planting_notes="Prefers riverbanks and moist areas"
        ),
        
        # BAMBOO SPECIES
        "Dendrocalamus strictus": Species(
            scientific_name="Dendrocalamus strictus",
            common_name="Solid Bamboo",
            family="Poaceae",
            native_regions=["All India except extreme arid"],
            forest_types=["All deciduous types"],
            canopy_layer="understory",
            growth_rate="fast",
            rainfall_min=600, rainfall_max=2000,
            temperature_min=15, temperature_max=40,
            soil_ph_min=5.5, soil_ph_max=7.5,
            elevation_min=0, elevation_max=1500,
            carbon_sequestration=150,
            ecological_value=["fast growth", "soil conservation", "economic"],
            planting_notes="Clump forming, flowers once in 30-40 years"
        ),
        
        # GROUND COVER
        "Ziziphus nummularia": Species(
            scientific_name="Ziziphus nummularia",
            common_name="Jhar Beri",
            family="Rhamnaceae",
            native_regions=["Rajasthan", "Gujarat", "Punjab"],
            forest_types=["Desert", "Tropical Thorn"],
            canopy_layer="ground",
            growth_rate="fast",
            rainfall_min=100, rainfall_max=500,
            temperature_min=5, temperature_max=48,
            soil_ph_min=7.0, soil_ph_max=8.5,
            elevation_min=0, elevation_max=600,
            carbon_sequestration=50,
            ecological_value=["fodder", "soil binder", "drought resistant"],
            planting_notes="Excellent for sand dune stabilization"
        )
    }
    
    return species_db

def get_species_by_region(region: str) -> List[Species]:
    """Get all species native to a specific region"""
    db = get_species_database()
    return [sp for sp in db.values() if region in sp.native_regions]

def get_species_by_forest_type(forest_type: str) -> List[Species]:
    """Get all species suitable for a specific forest type"""
    db = get_species_database()
    return [sp for sp in db.values() if forest_type in sp.forest_types]

def get_species_by_layer(layer: str) -> List[Species]:
    """Get all species belonging to a specific canopy layer"""
    db = get_species_database()
    return [sp for sp in db.values() if sp.canopy_layer == layer]

def filter_species_by_conditions(
    rainfall: float,
    temperature: float,
    soil_ph: float,
    elevation: float = None
) -> List[Species]:
    """Filter species based on environmental conditions"""
    db = get_species_database()
    suitable = []
    
    for species in db.values():
        if (species.rainfall_min <= rainfall <= species.rainfall_max and
            species.temperature_min <= temperature <= species.temperature_max and
            species.soil_ph_min <= soil_ph <= species.soil_ph_max):
            
            if elevation is not None:
                if species.elevation_min <= elevation <= species.elevation_max:
                    suitable.append(species)
            else:
                suitable.append(species)
    
    return suitable
