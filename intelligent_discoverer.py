import requests
import json
import time
from typing import Set, Dict, Any, List

# --- Configuration ---
# PASTE YOUR IUCN API TOKEN HERE
IUCN_API_TOKEN = "qEkCxqDZoNR4EompQejRqmjnsXRJYmV1HfK8" 

# A small, curated list of known invasive species in India. This can be expanded.
INVASIVE_SPECIES_BLACKLIST = {
    "Lantana camara",
    "Prosopis juliflora",
    "Parthenium hysterophorus",
    "Mikania micrantha",
    "Ageratina adenophora"
}

from adapters.gbif_client import get_species_in_area
from curation_helper import generate_species_template # We'll reuse the template generator

# --- Intelligent Pre-processing Functions ---

def get_iucn_status(species_name: str) -> str:
    """Fetches the conservation status for a species from the IUCN Red List API."""
    if not IUCN_API_TOKEN or IUCN_API_TOKEN == "YOUR_IUCN_API_TOKEN_HERE":
        return "Not Fetched (No API Token)"

    try:
        url = f"http://apiv3.iucnredlist.org/api/v3/species/{species_name}?token={IUCN_API_TOKEN}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('result'):
                return data['result'][0].get('category', 'Data Deficient')
    except requests.exceptions.RequestException as e:
        print(f"  - WARN: IUCN API request failed for {species_name}: {e}")
    
    return "Data Deficient"

def is_plant(species_name: str) -> bool:
    """Uses the GBIF API to verify if a species is in the Plantae kingdom."""
    try:
        url = f"https://api.gbif.org/v1/species/match?name={species_name}&kingdom=Plantae"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # Check if the match confidence is high and it's in the plant kingdom
            if data.get('matchType') != 'NONE' and data.get('kingdom') == 'Plantae':
                return True
    except requests.exceptions.RequestException:
        pass # Ignore network errors, will return False
    return False

# --- Main Discovery and Processing Logic ---

def run_intelligent_discovery(aoi_list: List[Dict], output_file: str):
    """
    Runs an intelligent discovery and pre-processing pipeline.
    """
    print("--- 🔬 Starting Intelligent Species Discovery ---")
    
    master_species_list = set()
    for aoi in aoi_list:
        print(f"\n--- Querying AOI: {aoi['name']} ---")
        species_found = get_species_in_area(aoi)
        if species_found:
            print(f"  - Found {len(species_found)} raw species names.")
            master_species_list.update(species_found)
    
    print(f"\n--- Discovery Complete ---")
    print(f"Found a total of {len(master_species_list)} unique raw names across all AOIs.")
    
    if not master_species_list:
        print("No species found. Exiting.")
        return

    print("\n--- 🧠 Starting Intelligent Pre-processing ---")
    enriched_templates = []
    sorted_species = sorted(list(master_species_list))
    
    for i, name in enumerate(sorted_species):
        print(f"({i+1}/{len(sorted_species)}) Processing: {name}")

        # 1. Filter: Is it a plant?
        if not is_plant(name):
            print(f"  - ❌ REJECTED: Not a confirmed plant species.")
            time.sleep(0.5) # Be kind to the API
            continue

        # 2. Filter: Is it invasive?
        if name in INVASIVE_SPECIES_BLACKLIST:
            print(f"  - ❌ REJECTED: Known invasive species.")
            continue

        # 3. Enrich: Get IUCN Status
        print(f"  - ✅ ACCEPTED: Verified plant.")
        print(f"  - Fetching IUCN status...")
        status = get_iucn_status(name)
        print(f"  - Status: {status}")

        # 4. Generate Enriched Template
        template = generate_species_template(name)
        template['conservation_status'] = status # Overwrite placeholder
        template['is_invasive'] = False # We already checked
        enriched_templates.append(template)
        time.sleep(1) # Rate limit to be kind to the IUCN API

    # 5. Save the final, enriched batch file
    with open(output_file, 'w') as f:
        json.dump(enriched_templates, f, indent=2)
        
    print(f"\n✅ SUCCESS: Saved {len(enriched_templates)} enriched templates to '{output_file}'.")
    print("➡️ Next step: Manually curate this file and then use the 'append' command.")

if __name__ == "__main__":
    # You can expand this list with more AOIs for a more comprehensive discovery
    aoi_definitions = [
        {
            "name": "Western Ghats (Agumbe)",
            "polygon_geojson": { "type": "Polygon", "coordinates": [[[75.0, 13.5], [75.2, 13.5], [75.2, 13.7], [75.0, 13.7], [75.0, 13.5]]]}
        },
        {
            "name": "Himalayas (Valley of Flowers)",
            "polygon_geojson": { "type": "Polygon", "coordinates": [[[79.5, 30.6], [79.7, 30.6], [79.7, 30.8], [79.5, 30.8], [79.5, 30.6]]]}
        }
    ]
    run_intelligent_discovery(aoi_list=aoi_definitions, output_file="curation_batch_intelligent.json")