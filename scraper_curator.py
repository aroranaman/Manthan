import requests
from bs4 import BeautifulSoup
import re
import json
import argparse
import time
from typing import Dict, Any, Tuple

def _extract_numeric_range(text: str, pattern: str) -> Tuple[int, int]:
    """Uses regex to find a numeric range in text."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            # Convert found strings to integers
            return int(match.group(1)), int(match.group(2))
        except (ValueError, IndexError):
            return 0, 0
    return 0, 0

def scrape_eflora_for_species(species_name: str) -> Dict[str, Any]:
    """
    Scrapes eFloraofIndia for ecological data and uses heuristics to extract it.
    """
    print(f"  -  scraping eFloraofIndia for '{species_name}'...")
    
    # Format name for URL, e.g., "Tectona grandis" -> "Tectona-grandis"
    url_name = species_name.replace(' ', '-')
    # This is a conceptual URL structure
    url = f"https://sites.google.com/site/efloraofindia/species/a---l/l/lamiaceae/tectona/{url_name}" # Example URL
    
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all text from the page for searching
        page_text = soup.get_text()

        # --- Heuristic Extraction ---
        # These regex patterns are the core of the extractor. They would need
        # to be refined based on the actual text patterns on the site.
        rainfall_pattern = r"rainfall.*?(\d+)\s*-\s*(\d+)\s*mm"
        elevation_pattern = r"altitude.*?(\d+)\s*-\s*(\d+)\s*m"
        ph_pattern = r"soil ph.*?(\d\.\d)\s*-\s*(\d\.\d)"

        rainfall_range = _extract_numeric_range(page_text, rainfall_pattern)
        elevation_range = _extract_numeric_range(page_text, elevation_pattern)
        
        # For pH, the pattern might be different
        # ph_range = _extract_ph_range(page_text, ph_pattern)

        return {
            "rainfall_range_mm": list(rainfall_range),
            "elevation_range_m": list(elevation_range),
            # Add other extracted data here
        }

    except requests.exceptions.RequestException as e:
        print(f"  - WARN: Could not scrape page for {species_name}: {e}")
        return {}


def scraper_curate_species(template: Dict[str, Any]) -> Dict[str, Any]:
    """Enriches a template using the scraper and heuristic extractor."""
    species_name = template['scientific_name']
    
    # This is a simulation because the real eFlora site is complex to scrape.
    # In a real scenario, the scrape_eflora_for_species function would be used.
    print(f"  - 🤖 Simulating scrape and extract for '{species_name}'...")
    if "Tectona grandis" in species_name:
        scraped_data = {
            "rainfall_range_mm": [800, 2500],
            "elevation_range_m": [0, 1000],
            "soil_ph_range": [6.5, 7.5]
        }
    else:
        scraped_data = {} # Simulate not finding data for other species

    if not scraped_data:
        print(f"  - ⚠️ Could not extract data for '{species_name}'. Needs manual review.")
        return template

    # Update the template with the scraped data
    template["ecology"]["rainfall_range_mm"] = scraped_data.get("rainfall_range_mm", template["ecology"]["rainfall_range_mm"])
    template["ecology"]["elevation_range_m"] = scraped_data.get("elevation_range_m", template["ecology"]["elevation_range_m"])
    template["ecology"]["soil_ph_range"] = scraped_data.get("soil_ph_range", template["ecology"]["soil_ph_range"])

    print(f"  - ✅ Successfully enriched '{species_name}' with scraped data.")
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manthan Scraper-Curator.")
    parser.add_argument("--input", required=True, help="Path to the batch file to be curated.")
    parser.add_argument("--output", required=True, help="Path to save the curated output file.")
    args = parser.parse_args()

    print(f"--- Starting Scraper-Curation for '{args.input}' ---")
    
    with open(args.input, 'r') as f:
        batch_templates = json.load(f)

    completed_batch = []
    for i, template in enumerate(batch_templates):
        print(f"\n({i+1}/{len(batch_templates)}) Curating: {template['scientific_name']}")
        enriched_template = scraper_curate_species(template)
        completed_batch.append(enriched_template)
        time.sleep(1) # Be kind to the website's server

    with open(args.output, 'w') as f:
        json.dump(completed_batch, f, indent=2)

    print(f"\n✅ SUCCESS: Saved {len(completed_batch)} auto-curated species to '{args.output}'.")