# FILE: src/processing/generate_ncf_training_data.py

import requests
import pandas as pd
from pathlib import Path
import time
import sys

# --- Robust Path Fix ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Fix ---

from src.core.geospatial_handler import GeospatialDataHandler # We reuse our GEE tool

# --- CONFIG ---
DATA_DIR = project_root / "src" / "data"
# A list of ~2500 native Indian plant species (you would build this list)
SPECIES_LIST_PATH = DATA_DIR / "indian_native_species.csv"
OUTPUT_PATH = DATA_DIR / "ncf_training_data.csv"
INDIA_COUNTRY_CODE = "IN"
MAX_RECORDS_PER_SPECIES = 100 # To keep the dataset balanced and manageable

def fetch_gbif_occurrences(species_name: str) -> list:
    """Queries the GBIF API for occurrence data for a given species in India."""
    base_url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "scientificName": species_name,
        "country": INDIA_COUNTRY_CODE,
        "hasCoordinate": "true",
        "limit": MAX_RECORDS_PER_SPECIES
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"  - GBIF API Error for '{species_name}': {e}")
        return []

def generate_data():
    """Generates a training dataset by combining GBIF and GEE data."""
    if not SPECIES_LIST_PATH.exists():
        print(f"Error: Species list not found at {SPECIES_LIST_PATH}")
        print("Please create a CSV with a 'scientific_name' column.")
        return

    species_to_query = pd.read_csv(SPECIES_LIST_PATH)['scientific_name'].tolist()
    
    print("Initializing GEE Handler...")
    gee_handler = GeospatialDataHandler()
    
    all_training_data = []
    
    print(f"Starting data generation for {len(species_to_query)} species...")
    for i, species_name in enumerate(species_to_query):
        print(f"[{i+1}/{len(species_to_query)}] Querying GBIF for '{species_name}'...")
        occurrences = fetch_gbif_occurrences(species_name)
        
        if not occurrences:
            continue
            
        for occ in occurrences:
            lon, lat = occ.get('decimalLongitude'), occ.get('decimalLatitude')
            if not all([lon, lat]):
                continue

            # Create a point AOI for GEE analysis
            aoi_geojson = {"geometry": {"type": "Point", "coordinates": [lon, lat]}}
            
            try:
                # Use our existing GEE tool to get site features for this location
                site_stats = gee_handler.comprehensive_site_analysis(
                    aoi=aoi_geojson,
                    start_date="2024-01-01",
                    end_date="2024-12-31"
                )['site_statistics']

                # Flatten the stats and add the species name
                site_stats['species_name'] = species_name
                all_training_data.append(site_stats)

            except Exception as e:
                print(f"  - GEE Error at ({lat:.2f}, {lon:.2f}): {e}")

        # Be respectful to the API
        time.sleep(1)

    df = pd.DataFrame(all_training_data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nData generation complete. {len(df)} records saved to:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    generate_data()