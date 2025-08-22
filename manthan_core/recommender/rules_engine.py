from manthan_core.schemas.site_fingerprint import SiteFingerprint
from manthan_core.schemas.species_rec import SpeciesRec, PlantingPlan
from manthan_core.schemas.aoi import AOI
from adapters.gbif_client import get_species_in_area
import json

def load_species_database():
    try:
        with open('assets/species_database.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: species_database.json not found in assets/.")
        return []

SPECIES_DB = load_species_database()

def recommend_species(fingerprint: SiteFingerprint, aoi: AOI) -> PlantingPlan:
    """Recommends species using a hybrid, data-driven approach with soft scoring."""
    
    candidate_species_names = get_species_in_area(aoi.polygon_geojson)
    
    if not candidate_species_names:
        print("WARNING: No species found via GBIF. Using entire local DB as fallback.")
        candidates = SPECIES_DB
    else:
        candidates = [s for s in SPECIES_DB if s['scientific_name'] in candidate_species_names]

    print(f"INFO: Starting with {len(candidates)} candidate species for filtering.")

    # --- Hard Filters ---
    suitable_species = []
    for species in candidates:
        min_rain, max_rain = species['ecology']['rainfall_range_mm']
        if not (min_rain <= fingerprint.rainfall_mm_year <= max_rain):
            continue

        min_ph, max_ph = species['ecology']['soil_ph_range']
        if not (min_ph <= fingerprint.soil_ph_estimate <= max_ph):
            continue
            
        suitable_species.append(species)

    print(f"INFO: {len(suitable_species)} species passed hard filters.")

    # --- Soft Scoring ---
    scored_species = []
    for species in suitable_species:
        # Calculate rainfall score (0-1)
        optimal_rain = species['ecology']['optimal_rainfall']
        rain_range = species['ecology']['rainfall_range_mm'][1] - species['ecology']['rainfall_range_mm'][0]
        rain_dist = abs(fingerprint.rainfall_mm_year - optimal_rain)
        rainfall_score = max(0, 1 - (rain_dist / (rain_range / 2)))

        # Calculate pH score (0-1)
        optimal_ph = species['ecology']['optimal_ph']
        ph_range = species['ecology']['soil_ph_range'][1] - species['ecology']['soil_ph_range'][0]
        ph_dist = abs(fingerprint.soil_ph_estimate - optimal_ph)
        ph_score = max(0, 1 - (ph_dist / (ph_range / 2)))
        
        # Combine scores (weighted average)
        final_score = (rainfall_score * 0.6) + (ph_score * 0.4)
        
        why = f"Good match for rainfall (score: {rainfall_score:.2f}) and soil pH (score: {ph_score:.2f})."
        
        rec = SpeciesRec(
            scientific_name=species['scientific_name'],
            common_name=species['common_name'],
            layer=species['traits']['layer'],
            native=True,
            why=why,
            score=final_score
        )
        scored_species.append(rec)

    # Sort by the new, more intelligent score
    scored_species.sort(key=lambda x: x.score, reverse=True)
    
    plan = PlantingPlan(fingerprint=fingerprint, recommendations=scored_species[:10])
    print("✅ INFO: Successfully generated dynamic species recommendations with soft scoring.")
    return plan