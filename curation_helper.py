import requests
import json
import argparse
from typing import List, Dict, Any

# --- Configuration ---
IBP_API_URL = "https://indiabiodiversity.org/species/show_by_name"
DB_PATH = 'assets/species_database.json'

# --- Mode 1: Generate Templates ---

def get_species_data_from_ibp(scientific_name: str) -> dict:
    """Simulates fetching data for a species from the India Biodiversity Portal."""
    print(f"INFO: Searching IBP for '{scientific_name}'...")
    # This is a simulation. A real implementation would involve a more complex API call or web scraping.
    simulated_data = {
        "common_names": [f"_ADD_COMMON_NAME_FOR_{scientific_name.replace(' ', '_')}_"]
    }
    return simulated_data

def generate_species_template(scientific_name: str) -> dict:
    """Generates a pre-filled JSON template for a species for human curation."""
    ibp_data = get_species_data_from_ibp(scientific_name)
    common_name = ibp_data.get("common_names", ["_ADD_COMMON_NAME_"])[0]

    return {
        "scientific_name": scientific_name,
        "common_name": common_name,
        "is_invasive": False,
        "conservation_status": "_ADD_STATUS_",
        "traits": {"layer": "_ADD_LAYER_", "growth_form": "_ADD_FORM_", "functional_role": "_ADD_ROLE_"},
        "ecology": {
            "native_to_zones": ["_ADD_ZONES_"],
            "rainfall_range_mm": [0, 0], "optimal_rainfall": 0,
            "elevation_range_m": [0, 0],
            "soil_ph_range": [0.0, 0.0], "optimal_ph": 0.0
        }
    }

def handle_generate(species_names: List[str]):
    """Handles the 'generate' command."""
    print("--- Generating Curation Templates ---")
    all_templates = [generate_species_template(name) for name in species_names]
    print("\n--- COPY, COMPLETE, AND SAVE THE JSON BELOW TO A NEW FILE (e.g., 'new_species.json') ---")
    print(json.dumps(all_templates, indent=2))

# --- Mode 2: Append to Database ---

def handle_append(input_file: str):
    """Handles the 'append' command."""
    print(f"--- Appending data from '{input_file}' to '{DB_PATH}' ---")
    
    # 1. Load the existing database
    try:
        with open(DB_PATH, 'r') as f:
            db_data = json.load(f)
    except FileNotFoundError:
        db_data = []
    
    existing_names = {species['scientific_name'] for species in db_data}
    print(f"INFO: Loaded {len(db_data)} species from the existing database.")

    # 2. Load the new, curated species data
    try:
        with open(input_file, 'r') as f:
            new_data = json.load(f)
        if not isinstance(new_data, list):
            raise ValueError("Input file must contain a JSON list of species.")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Could not read or parse input file '{input_file}': {e}")
        return

    # 3. Append new, non-duplicate species
    appended_count = 0
    for species in new_data:
        name = species.get('scientific_name')
        if not name:
            print(f"WARNING: Skipping record with no scientific name: {species}")
            continue
        
        if name not in existing_names:
            db_data.append(species)
            existing_names.add(name)
            appended_count += 1
        else:
            print(f"INFO: Skipping duplicate species '{name}'.")

    # 4. Write the updated database back to the file
    if appended_count > 0:
        with open(DB_PATH, 'w') as f:
            json.dump(db_data, f, indent=2)
        print(f"✅ SUCCESS: Appended {appended_count} new species. Database now contains {len(db_data)} species.")
    else:
        print("INFO: No new species were appended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manthan Species Database Curation Helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-parser for the "generate" command
    parser_generate = subparsers.add_parser("generate", help="Generate templates for new species.")
    parser_generate.add_argument("names", nargs='+', help="The scientific names of the species to generate templates for.")

    # Sub-parser for the "append" command
    parser_append = subparsers.add_parser("append", help="Append curated species data to the main database.")
    parser_append.add_argument("--input", required=True, help="Path to the JSON file containing the new, curated species data.")

    args = parser.parse_args()

    if args.command == "generate":
        handle_generate(args.names)
    elif args.command == "append":
        handle_append(args.input)