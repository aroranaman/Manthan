import requests
from typing import Set, Dict, Any
from shapely.geometry import shape

# GBIF API endpoint
GBIF_API_URL = "https://api.gbif.org/v1/occurrence/search"

def get_species_in_area(aoi_geojson: Dict[str, Any]) -> Set[str]:
    """
    Queries the GBIF API to find plant species occurrences within a given AOI
    using a robust occurrence search.
    """
    print(f"INFO: Querying GBIF for species in AOI...")
    
    try:
        # Handle both the full AOI dict and just the geojson part
        geojson_poly = aoi_geojson.get("polygon_geojson", aoi_geojson)
        polygon = shape(geojson_poly)
        wkt_geometry = polygon.wkt
    except Exception as e:
        print(f"ERROR: Could not process GeoJSON for GBIF query: {e}")
        return set()

    params = {
        'geometry': wkt_geometry,
        'country': 'IN',
        'kingdom': 'Plantae',
        'hasCoordinate': 'true',
        'limit': 200  # Get up to 200 raw occurrence records
    }

    try:
        response = requests.get(GBIF_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        species_found = set()
        # Iterate through the raw results and extract the species name
        for record in data.get('results', []):
            species_name = record.get('species')
            if species_name:
                species_found.add(species_name)
        
        if not species_found:
            print("WARNING: No species found in GBIF for this specific AOI.")
            return set()
            
        print(f"✅ INFO: Found {len(species_found)} unique species in the AOI via GBIF.")
        return species_found

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to connect to GBIF API: {e}")
        return set()