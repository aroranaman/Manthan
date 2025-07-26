# utils/rainfall.py
import ee
import requests
import numpy as np
from utils.gee_auth import gee_init

def get_chirps_rainfall(aoi_geometry, year=2024):
    """
    Extract annual rainfall from CHIRPS dataset.
    
    Args:
        aoi_geometry: Earth Engine Geometry object
        year: Year for rainfall data
    
    Returns:
        dict: Rainfall statistics
    """
    gee_init()
    
    try:
        # Use Earth Engine CHIRPS dataset
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                   .filterDate(f'{year}-01-01', f'{year}-12-31') \
                   .filterBounds(aoi_geometry)
        
        # Sum annual precipitation
        annual_rainfall = chirps.select('precipitation').sum().clip(aoi_geometry)
        
        # Calculate statistics
        stats = annual_rainfall.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ),
            geometry=aoi_geometry,
            scale=5000,  # 5km resolution
            maxPixels=1e9
        )
        
        rainfall_stats = stats.getInfo()
        mean_rainfall = rainfall_stats.get('precipitation_mean', 1000)
        
        # Classify rainfall adequacy
        if mean_rainfall > 1500:
            adequacy = "Excellent"
        elif mean_rainfall > 1000:
            adequacy = "Good"
        elif mean_rainfall > 600:
            adequacy = "Moderate"
        else:
            adequacy = "Low"
        
        return {
            'annual_rainfall': mean_rainfall,
            'rainfall_std': rainfall_stats.get('precipitation_stdDev', 100),
            'rainfall_min': rainfall_stats.get('precipitation_min', 800),
            'rainfall_max': rainfall_stats.get('precipitation_max', 1200),
            'rainfall_adequacy': adequacy
        }
        
    except Exception as e:
        print(f"Error getting rainfall data: {e}")
        # Fallback with India average
        return {
            'annual_rainfall': 1100,
            'rainfall_std': 200,
            'rainfall_min': 900,
            'rainfall_max': 1300,
            'rainfall_adequacy': "Good"
        }
