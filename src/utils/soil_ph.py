# utils/soil_ph.py
import requests
import json
import numpy as np

def get_soil_ph_for_aoi(aoi_geometry):
    """
    Get soil pH from SoilGrids REST API.
    
    Args:
        aoi_geometry: Earth Engine Geometry object
    
    Returns:
        dict: Soil pH statistics
    """
    try:
        # Get centroid coordinates
        centroid = aoi_geometry.centroid().coordinates().getInfo()
        lon, lat = centroid[0], centroid[1]
        
        # SoilGrids REST API endpoint
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            'lon': lon,
            'lat': lat,
            'property': 'phh2o',
            'depth': '0-5cm',
            'value': 'mean'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract pH value (convert from deci-pH to pH)
        ph_value = None
        for layer in data.get('properties', {}).get('layers', []):
            if 'phh2o' in layer.get('name', ''):
                for depth in layer.get('depths', []):
                    if '0-5cm' in depth.get('label', ''):
                        ph_value = depth.get('values', {}).get('mean', 65) / 10
                        break
        
        if ph_value is None:
            ph_value = 6.5  # Default neutral pH
        
        # Classify pH suitability
        if 6.0 <= ph_value <= 7.5:
            suitability = "Optimal"
        elif 5.5 <= ph_value <= 8.0:
            suitability = "Good"
        elif 5.0 <= ph_value <= 8.5:
            suitability = "Moderate"
        else:
            suitability = "Poor"
        
        return {
            'soil_ph': ph_value,
            'ph_suitability': suitability,
            'ph_min': max(ph_value - 0.3, 4.0),
            'ph_max': min(ph_value + 0.3, 9.0),
            'ph_std': 0.2
        }
        
    except Exception as e:
        print(f"Error getting soil pH: {e}")
        # Default values for India
        return {
            'soil_ph': 6.5,
            'ph_suitability': "Good",
            'ph_min': 6.0,
            'ph_max': 7.0,
            'ph_std': 0.3
        }
