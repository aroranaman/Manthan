# FILE: src/processing/build_species_database.py
import requests
import json
import time
from pathlib import Path

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual IUCN API token
API_TOKEN = "qEkCxqDZoNR4EompQejRqmjnsXRJYmV1HfK8" 
BASE_URL = "https://api.iucnredlist.org/"

# --- INPUT: A base list of species to look up ---
# In a real project, this list would come from a source like India Biodiversity Portal or ENVIS
SPECIES_TO_QUERY = {
  "North India": {
    "Uttar Pradesh": ["Mangifera indica", "Shorea robusta", "Azadirachta indica"],
    "Rajasthan": ["Prosopis cineraria", "Tecomella undulata", "Withania somnifera"],
    "Uttarakhand": ["Rhododendron arboreum", "Taxus wallichiana"]
  },
  "South India": {
      "Karnataka": ["Santalum album", "Pterocarpus marsupium"],
      "Tamil Nadu": ["Pterocarpus santalinus", "Tamarindus indica"]
  },
  "Northeast India": {
      "Assam": ["Aquilaria malaccensis", "Dipterocarpus retusus"]
  }
}

# --- OUTPUT ---
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "api_generated_species_database.json"

def get_conservation_status(species_name: str) -> str:
    """Queries the IUCN API for the conservation status of a single species."""
    if not API_TOKEN or "YOUR_IUCN_API_TOKEN" in API_TOKEN:
        return "API Token Not Set"
        
    endpoint = f"/species/{species_name.lower().replace(' ', '%20')}"
    params = {"token": API_TOKEN}
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        # The 'result' list might be empty if the species is not found or has no assessment
        if not data.get('result'):
            return "Not Evaluated"
            
        # Return the category code (e.g., 'LC', 'EN', 'VU')
        return data['result'][0].get('category', 'Data Deficient')
        
    except requests.exceptions.RequestException as e:
        print(f"  - API Error for '{species_name}': {e}")
        # Return a status that indicates an error occurred
        return "API Error"

def build_database():
    """Iterates through regions and species to build a comprehensive JSON database."""
    print("Starting to build species database from IUCN API...")
    full_database = {}
    
    # Mapping of IUCN codes to human-readable status
    status_map = {
        "LC": "Least Concern", "NT": "Near Threatened", "VU": "Vulnerable",
        "EN": "Endangered", "CR": "Critically Endangered", "EW": "Extinct in the Wild",
        "EX": "Extinct", "DD": "Data Deficient", "NE": "Not Evaluated"
    }

    for region, states in SPECIES_TO_QUERY.items():
        full_database[region] = {}
        print(f"\nProcessing Region: {region}")
        for state, species_list in states.items():
            full_database[region][state] = []
            print(f"- State: {state}")
            for species in species_list:
                print(f"  - Querying: {species}...")
                status_code = get_conservation_status(species)
                
                species_entry = {
                    "scientific_name": species,
                    "common_name": "N/A", # This could be enriched from another source
                    "type": "N/A",      # This could be enriched from another source
                    "conservation_status": status_map.get(status_code, status_code),
                    "iucn_code": status_code,
                    "is_invasive": False # Default, could be enriched from an invasive species DB
                }
                full_database[region][state].append(species_entry)
                
                # Be a responsible API user, wait a second between requests
                time.sleep(1)

    # --- Save the final enriched database ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(full_database, f, indent=4)
        
    print(f"\n✅ Database generation complete!")
    print(f"Data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_database()