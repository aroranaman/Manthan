import ee
import json
import argparse
import requests
import numpy as np
import signal
from typing import List, Dict, Any

from adapters.gee_connector import initialize_gee

# --- Configuration ---
GBIF_API_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_SPECIES_MATCH_URL = "https://api.gbif.org/v1/species/match"
MIN_OCCURRENCES = 5
MAX_RECORDS_TO_FETCH = 500  # Reduced for faster processing
SPECIES_TIMEOUT_SECONDS = 90 # Max time to spend on one species

# --- GEE Image Assets ---
try:
    if initialize_gee():
        PRECIPITATION_IMG = ee.Image("WORLDCLIM/V1/BIO").select('bio12')
        SOIL_PH_IMG = ee.Image("projects/soilgrids-isric/phh2o_mean").select('phh2o_0-5cm_mean')
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize GEE. Trait extractor cannot run. {e}")
    PRECIPITATION_IMG = None
    SOIL_PH_IMG = None

# --- Timeout Handler ---
class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

# --- Helper Functions ---

def is_plant(species_name: str) -> bool:
    """A more robust function to verify if a species is in the Plantae kingdom."""
    try:
        params = {'name': species_name, 'verbose': 'false'}
        response = requests.get(GBIF_SPECIES_MATCH_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('matchType') != 'NONE' and data.get('kingdom') == 'Plantae':
            return True
    except requests.exceptions.RequestException as e:
        print(f"  - WARN: is_plant check failed for {species_name}: {e}")
        return False
    return False

def get_gbif_occurrences(species_name: str) -> List[Dict[str, float]]:
    """Gets occurrence coordinates for a species, with a cap and progress updates."""
    all_coords = []
    params = {
        'scientificName': species_name,
        'country': 'IN',
        'hasCoordinate': 'true',
        'limit': 200,
        'offset': 0
    }
    page = 1
    while len(all_coords) < MAX_RECORDS_TO_FETCH:
        try:
            print(f"    - Fetching page {page}...", end='\r')
            response = requests.get(GBIF_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            for rec in results:
                if 'decimalLatitude' in rec and 'decimalLongitude' in rec:
                    all_coords.append({
                        'lat': rec['decimalLatitude'],
                        'lon': rec['decimalLongitude']
                    })
            if data['endOfRecords']:
                break
            params['offset'] += 200
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"  - WARN: GBIF request failed: {e}")
            break
    print("") # Newline after progress updates
    return all_coords[:MAX_RECORDS_TO_FETCH]

def sample_environmental_data(coords: List[Dict[str, float]]) -> Dict[str, List[float]]:
    # ... (rest of the function is unchanged)
    if not PRECIPITATION_IMG or not SOIL_PH_IMG:
        return {"precipitation": [], "soil_ph": []}

    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(c['lon'], c['lat'])) for c in coords
    ])

    def sample_point(point):
        precip = PRECIPITATION_IMG.reduceRegion(ee.Reducer.first(), point.geometry(), 1000).get('bio12')
        soil_ph = SOIL_PH_IMG.reduceRegion(ee.Reducer.first(), point.geometry(), 1000).get('phh2o_0-5cm_mean')
        return point.set({'precipitation': precip, 'soil_ph': soil_ph})

    try:
        sampled_data = points.map(sample_point).getInfo()
        
        precip_values = [f['properties']['precipitation'] for f in sampled_data['features'] if f['properties'].get('precipitation') is not None]
        ph_values = [f['properties']['soil_ph'] / 10.0 for f in sampled_data['features'] if f['properties'].get('soil_ph') is not None]
        
        return {"precipitation": precip_values, "soil_ph": ph_values}
    except Exception as e:
        print(f"  - WARN: GEE sampling failed: {e}")
        return {"precipitation": [], "soil_ph": []}

# --- Main Logic ---

def extract_traits_for_species(species_name: str) -> Dict[str, Any]:
    """Main function to extract ecological traits for a single species."""
    print(f"Processing: {species_name}")
    
    if not is_plant(species_name):
        print(f"  - ⚠️ NOT A PLANT. Skipping.")
        return None

    print("  - Fetching occurrence data from GBIF...")
    coords = get_gbif_occurrences(species_name)
    if len(coords) < MIN_OCCURRENCES:
        print(f"  - ⚠️ INSUFFICIENT DATA: Found only {len(coords)} occurrences. Skipping.")
        return None
    print(f"  - Found {len(coords)} occurrence records.")

    print("  - Sampling environmental data from GEE...")
    env_data = sample_environmental_data(coords)
    
    if not env_data['precipitation'] or not env_data['soil_ph']:
        print(f"  - ⚠️ GEE SAMPLING FAILED. Skipping.")
        return None

    print("  - Calculating ecological niche...")
    rainfall_range = [int(np.percentile(env_data['precipitation'], 5)), int(np.percentile(env_data['precipitation'], 95))]
    optimal_rainfall = int(np.median(env_data['precipitation']))
    
    ph_range = [round(np.percentile(env_data['soil_ph'], 5), 1), round(np.percentile(env_data['soil_ph'], 95), 1)]
    optimal_ph = round(np.median(env_data['soil_ph']), 1)

    return {
        "scientific_name": species_name,
        "common_name": f"_CURATED_{species_name.replace(' ', '_')}_",
        "is_invasive": False,
        "conservation_status": "_CURATED_",
        "traits": {"layer": "_CURATED_", "growth_form": "_CURATED_", "functional_role": "_CURATED_"},
        "ecology": {
            "native_to_zones": ["_CURATED_"],
            "rainfall_range_mm": rainfall_range,
            "optimal_rainfall": optimal_rainfall,
            "elevation_range_m": [0, 0],
            "soil_ph_range": ph_range,
            "optimal_ph": optimal_ph
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manthan Trait Extractor.")
    parser.add_argument("--input", required=True, help="Path to the file with species names.")
    parser.add_argument("--output", required=True, help="Path to save the extracted traits.")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        species_to_process = [s['scientific_name'] for s in json.load(f)]

    extracted_traits = []
    for name in species_to_process:
        signal.alarm(SPECIES_TIMEOUT_SECONDS) # Set the timeout for this species
        try:
            traits = extract_traits_for_species(name)
            if traits:
                extracted_traits.append(traits)
        except TimeoutException:
            print(f"  - ⚠️ TIMEOUT: Processing for '{name}' took too long. Skipping.")
        finally:
            signal.alarm(0) # Disable the alarm
    
    with open(args.output, 'w') as f:
        json.dump(extracted_traits, f, indent=2)

    print(f"\n✅ SUCCESS: Extracted traits for {len(extracted_traits)} species and saved to '{args.output}'.")