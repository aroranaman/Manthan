import json
from adapters.gbif_client import get_species_in_area
from curation_helper import generate_species_template

# --- 1. Define Representative AOIs Across India ---
# A sample of diverse ecological zones. We can expand this list significantly.
REPRESENTATIVE_AOIS = [
    # --- Himalayan Region ---
    {"name": "Great Himalayan NP, Himachal", "polygon_geojson": {"type": "Polygon", "coordinates": [[[77.4, 31.8], [77.6, 31.8], [77.6, 32.0], [77.4, 32.0], [77.4, 31.8]]]}},
    {"name": "Valley of Flowers, Uttarakhand", "polygon_geojson": {"type": "Polygon", "coordinates": [[[79.5, 30.6], [79.7, 30.6], [79.7, 30.8], [79.5, 30.8], [79.5, 30.6]]]}},
    {"name": "Khangchendzonga NP, Sikkim", "polygon_geojson": {"type": "Polygon", "coordinates": [[[88.2, 27.6], [88.4, 27.6], [88.4, 27.8], [88.2, 27.8], [88.2, 27.6]]]}},
    {"name": "Dachigam NP, Jammu & Kashmir", "polygon_geojson": {"type": "Polygon", "coordinates": [[[75.0, 34.1], [75.2, 34.1], [75.2, 34.3], [75.0, 34.3], [75.0, 34.1]]]}},
    {"name": "Nanda Devi Biosphere, Uttarakhand", "polygon_geojson": {"type": "Polygon", "coordinates": [[[79.8, 30.3], [80.0, 30.3], [80.0, 30.5], [79.8, 30.5], [79.8, 30.3]]]}},

    # --- Western Ghats ---
    {"name": "Agumbe Rainforest, Karnataka", "polygon_geojson": {"type": "Polygon", "coordinates": [[[75.0, 13.5], [75.2, 13.5], [75.2, 13.7], [75.0, 13.7], [75.0, 13.5]]]}},
    {"name": "Silent Valley NP, Kerala", "polygon_geojson": {"type": "Polygon", "coordinates": [[[76.4, 11.0], [76.6, 11.0], [76.6, 11.2], [76.4, 11.2], [76.4, 11.0]]]}},
    {"name": "Mhadei Wildlife Sanctuary, Goa", "polygon_geojson": {"type": "Polygon", "coordinates": [[[74.1, 15.5], [74.3, 15.5], [74.3, 15.7], [74.1, 15.7], [74.1, 15.5]]]}},
    {"name": "Bhadra Tiger Reserve, Karnataka", "polygon_geojson": {"type": "Polygon", "coordinates": [[[75.5, 13.4], [75.7, 13.4], [75.7, 13.6], [75.5, 13.6], [75.5, 13.4]]]}},
    {"name": "Koyna Wildlife Sanctuary, Maharashtra", "polygon_geojson": {"type": "Polygon", "coordinates": [[[73.7, 17.5], [73.9, 17.5], [73.9, 17.7], [73.7, 17.7], [73.7, 17.5]]]}},

    # --- Eastern Ghats ---
    {"name": "Araku Valley, Andhra Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[82.8, 18.2], [83.0, 18.2], [83.0, 18.4], [82.8, 18.4], [82.8, 18.2]]]}},
    {"name": "Simlipal NP, Odisha", "polygon_geojson": {"type": "Polygon", "coordinates": [[[86.2, 21.8], [86.4, 21.8], [86.4, 22.0], [86.2, 22.0], [86.2, 21.8]]]}},
    {"name": "Nallamala Hills, Andhra Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.7, 15.8], [78.9, 15.8], [78.9, 16.0], [78.7, 16.0], [78.7, 15.8]]]}},
    {"name": "Kolli Hills, Tamil Nadu", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.3, 11.2], [78.5, 11.2], [78.5, 11.4], [78.3, 11.4], [78.3, 11.2]]]}},
    {"name": "Gandhamardan Hills, Odisha", "polygon_geojson": {"type": "Polygon", "coordinates": [[[82.8, 20.8], [83.0, 20.8], [83.0, 21.0], [82.8, 21.0], [82.8, 20.8]]]}},

    # --- Arid & Semi-Arid Regions ---
    {"name": "Thar Desert (Jaisalmer), Rajasthan", "polygon_geojson": {"type": "Polygon", "coordinates": [[[70.8, 26.8], [71.0, 26.8], [71.0, 27.0], [70.8, 27.0], [70.8, 26.8]]]}},
    {"name": "Rann of Kutch, Gujarat", "polygon_geojson": {"type": "Polygon", "coordinates": [[[70.5, 23.5], [70.7, 23.5], [70.7, 23.7], [70.5, 23.7], [70.5, 23.5]]]}},
    {"name": "Sariska Tiger Reserve, Rajasthan", "polygon_geojson": {"type": "Polygon", "coordinates": [[[76.3, 27.2], [76.5, 27.2], [76.5, 27.4], [76.3, 27.4], [76.3, 27.2]]]}},
    {"name": "Velavadar NP, Gujarat", "polygon_geojson": {"type": "Polygon", "coordinates": [[[72.0, 22.0], [72.2, 22.0], [72.2, 22.2], [72.0, 22.2], [72.0, 22.0]]]}},
    {"name": "Deccan Thorn Scrub, Maharashtra", "polygon_geojson": {"type": "Polygon", "coordinates": [[[75.0, 18.0], [75.2, 18.0], [75.2, 18.2], [75.0, 18.2], [75.0, 18.0]]]}},

    # --- Central Indian Highlands ---
    {"name": "Kanha NP, Madhya Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[80.5, 22.2], [80.7, 22.2], [80.7, 22.4], [80.5, 22.4], [80.5, 22.2]]]}},
    {"name": "Bandhavgarh NP, Madhya Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[81.0, 23.7], [81.2, 23.7], [81.2, 23.9], [81.0, 23.9], [81.0, 23.7]]]}},
    {"name": "Pachmarhi Biosphere, Madhya Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.3, 22.4], [78.5, 22.4], [78.5, 22.6], [78.3, 22.6], [78.3, 22.4]]]}},
    {"name": "Tadoba Andhari Tiger Reserve, Maharashtra", "polygon_geojson": {"type": "Polygon", "coordinates": [[[79.2, 20.3], [79.4, 20.3], [79.4, 20.5], [79.2, 20.5], [79.2, 20.3]]]}},
    {"name": "Satpura NP, Madhya Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.0, 22.5], [78.2, 22.5], [78.2, 22.7], [78.0, 22.7], [78.0, 22.5]]]}},

    # --- Indo-Gangetic Plain ---
    {"name": "Dudhwa NP, Uttar Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[80.5, 28.5], [80.7, 28.5], [80.7, 28.7], [80.5, 28.7], [80.5, 28.5]]]}},
    {"name": "Jim Corbett NP, Uttarakhand", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.8, 29.5], [79.0, 29.5], [79.0, 29.7], [78.8, 29.7], [78.8, 29.5]]]}},
    {"name": "Valmiki NP, Bihar", "polygon_geojson": {"type": "Polygon", "coordinates": [[[84.0, 27.3], [84.2, 27.3], [84.2, 27.5], [84.0, 27.5], [84.0, 27.3]]]}},
    {"name": "Hastinapur WLS, Uttar Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.0, 29.1], [78.2, 29.1], [78.2, 29.3], [78.0, 29.3], [78.0, 29.1]]]}},
    {"name": "Rajaji NP, Uttarakhand", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.1, 30.0], [78.3, 30.0], [78.3, 30.2], [78.1, 30.2], [78.1, 30.0]]]}},

    # --- Northeast India ---
    {"name": "Kaziranga NP, Assam", "polygon_geojson": {"type": "Polygon", "coordinates": [[[93.2, 26.6], [93.4, 26.6], [93.4, 26.8], [93.2, 26.8], [93.2, 26.6]]]}},
    {"name": "Namdapha NP, Arunachal Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[96.2, 27.4], [96.4, 27.4], [96.4, 27.6], [96.2, 27.6], [96.2, 27.4]]]}},
    {"name": "Keibul Lamjao NP, Manipur", "polygon_geojson": {"type": "Polygon", "coordinates": [[[93.8, 24.5], [94.0, 24.5], [94.0, 24.7], [93.8, 24.7], [93.8, 24.5]]]}},
    {"name": "Balpakram NP, Meghalaya", "polygon_geojson": {"type": "Polygon", "coordinates": [[[90.7, 25.3], [90.9, 25.3], [90.9, 25.5], [90.7, 25.5], [90.7, 25.3]]]}},
    {"name": "Nokrek Ridge NP, Meghalaya", "polygon_geojson": {"type": "Polygon", "coordinates": [[[90.2, 25.4], [90.4, 25.4], [90.4, 25.6], [90.2, 25.6], [90.2, 25.4]]]}},

    # --- Coastal & Mangrove Areas ---
    {"name": "Sundarbans NP, West Bengal", "polygon_geojson": {"type": "Polygon", "coordinates": [[[88.8, 21.8], [89.0, 21.8], [89.0, 22.0], [88.8, 22.0], [88.8, 21.8]]]}},
    {"name": "Bhitarkanika NP, Odisha", "polygon_geojson": {"type": "Polygon", "coordinates": [[[86.8, 20.6], [87.0, 20.6], [87.0, 20.8], [86.8, 20.8], [86.8, 20.6]]]}},
    {"name": "Gulf of Mannar Marine NP, Tamil Nadu", "polygon_geojson": {"type": "Polygon", "coordinates": [[[79.1, 9.1], [79.3, 9.1], [79.3, 9.3], [79.1, 9.3], [79.1, 9.1]]]}},
    {"name": "Pichavaram Mangroves, Tamil Nadu", "polygon_geojson": {"type": "Polygon", "coordinates": [[[79.7, 11.4], [79.9, 11.4], [79.9, 11.6], [79.7, 11.6], [79.7, 11.4]]]}},
    {"name": "Coringa WLS, Andhra Pradesh", "polygon_geojson": {"type": "Polygon", "coordinates": [[[82.2, 16.7], [82.4, 16.7], [82.4, 16.9], [82.2, 16.9], [82.2, 16.7]]]}},

    # --- Islands ---
    {"name": "Saddle Peak NP, Andaman", "polygon_geojson": {"type": "Polygon", "coordinates": [[[92.9, 13.1], [93.1, 13.1], [93.1, 13.3], [92.9, 13.3], [92.9, 13.1]]]}},
    {"name": "Campbell Bay NP, Nicobar", "polygon_geojson": {"type": "Polygon", "coordinates": [[[93.8, 7.0], [94.0, 7.0], [94.0, 7.2], [93.8, 7.2], [93.8, 7.0]]]}},
    {"name": "Mount Harriet NP, Andaman", "polygon_geojson": {"type": "Polygon", "coordinates": [[[92.7, 11.8], [92.9, 11.8], [92.9, 12.0], [92.7, 12.0], [92.7, 11.8]]]}},
    {"name": "Rani Jhansi Marine NP, Andaman", "polygon_geojson": {"type": "Polygon", "coordinates": [[[92.6, 12.2], [92.8, 12.2], [92.8, 12.4], [92.6, 12.4], [92.6, 12.2]]]}},
    {"name": "Galathea NP, Nicobar", "polygon_geojson": {"type": "Polygon", "coordinates": [[[93.8, 7.2], [94.0, 7.2], [94.0, 7.4], [93.8, 7.4], [93.8, 7.2]]]}},

    # --- Deccan Plateau ---
    {"name": "Pench NP, Maharashtra", "polygon_geojson": {"type": "Polygon", "coordinates": [[[79.0, 21.6], [79.2, 21.6], [79.2, 21.8], [79.0, 21.8], [79.0, 21.6]]]}},
    {"name": "Melghat Tiger Reserve, Maharashtra", "polygon_geojson": {"type": "Polygon", "coordinates": [[[77.1, 21.3], [77.3, 21.3], [77.3, 21.5], [77.1, 21.5], [77.1, 21.3]]]}},
    {"name": "Nagarjunasagar-Srisailam Tiger Reserve, AP", "polygon_geojson": {"type": "Polygon", "coordinates": [[[78.8, 16.2], [79.0, 16.2], [79.0, 16.4], [78.8, 16.4], [78.8, 16.2]]]}},
    {"name": "Bandipur NP, Karnataka", "polygon_geojson": {"type": "Polygon", "coordinates": [[[76.5, 11.6], [76.7, 11.6], [76.7, 11.8], [76.5, 11.8], [76.5, 11.6]]]}},
    {"name": "Mudumalai NP, Tamil Nadu", "polygon_geojson": {"type": "Polygon", "coordinates": [[[76.5, 11.5], [76.7, 11.5], [76.7, 11.7], [76.5, 11.7], [76.5, 11.5]]]}},
]

def run_batch_discovery(output_file: str):
    """
    Runs the discovery process across all representative AOIs and generates
    a single large file of curation templates.
    """
    print("--- Starting Batch Species Discovery ---")
    
    master_species_list = set()
    
    for aoi in REPRESENTATIVE_AOIS:
        print(f"\n--- Querying AOI: {aoi['name']} ---")
        species_found = get_species_in_area(aoi)
        if species_found:
            print(f"Found {len(species_found)} species.")
            master_species_list.update(species_found)
    
    print(f"\n--- Discovery Complete ---")
    print(f"Found a total of {len(master_species_list)} unique species across all AOIs.")
    
    if not master_species_list:
        print("No species found. Exiting.")
        return

    print("\n--- Generating Curation Templates for All Discovered Species ---")
    all_templates = []
    for i, name in enumerate(sorted(list(master_species_list))):
        print(f"({i+1}/{len(master_species_list)}) Generating template for: {name}")
        template = generate_species_template(name)
        all_templates.append(template)
        
    # Save the large batch of templates to the specified output file
    with open(output_file, 'w') as f:
        json.dump(all_templates, f, indent=2)
        
    print(f"\n✅ SUCCESS: Saved {len(all_templates)} templates to '{output_file}'.")
    print("➡️ Next step: Manually curate this file and then use the 'append' command in curation_helper.py.")

if __name__ == "__main__":
    run_batch_discovery(output_file="curation_batch_1.json")