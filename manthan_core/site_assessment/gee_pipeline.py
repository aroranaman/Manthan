# manthan_core/site_assessment/gee_pipeline.py

import ee
import requests
from typing import Dict, Any

from manthan_core.schemas.aoi import AOI
from manthan_core.schemas.site_fingerprint import SiteFingerprint, Stats
from adapters.gee_connector import initialize_gee

# --- Helper Functions (Consolidated from your utils) ---

def _get_sentinel2_ndvi_stats(aoi_geometry: ee.Geometry) -> Dict[str, Any]:
    """Computes median NDVI statistics from Sentinel-2."""
    try:
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(aoi_geometry)
                        .filterDate('2023-01-01', '2023-12-31') # Using a recent full year
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        def calculate_ndvi(image):
            return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        ndvi_median = s2_collection.map(calculate_ndvi).median().clip(aoi_geometry)
        
        stats = ndvi_median.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.median(), "", True).combine(ee.Reducer.stdDev(), "", True),
            geometry=aoi_geometry,
            scale=30, # 30m is a reasonable scale for this analysis
            maxPixels=1e9
        ).getInfo()
        
        return {
            "mean": stats.get('NDVI_mean', 0),
            "median": stats.get('NDVI_median', 0),
            "std_dev": stats.get('NDVI_stdDev', 0)
        }
    except Exception as e:
        print(f"WARNING: Could not compute NDVI stats: {e}. Returning default values.")
        return {"mean": 0, "median": 0, "std_dev": 0}

def _get_chirps_rainfall(aoi_geometry: ee.Geometry) -> float:
    """Extracts mean annual rainfall from CHIRPS."""
    try:
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate('2023-01-01', '2023-12-31')
        annual_rainfall = chirps.sum().clip(aoi_geometry)
        
        stats = annual_rainfall.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_geometry,
            scale=5000,
            maxPixels=1e9
        ).getInfo()
        
        return stats.get('precipitation', 0)
    except Exception as e:
        print(f"WARNING: Could not compute rainfall: {e}. Returning default value.")
        return 0

def _get_soil_ph(aoi_geometry: ee.Geometry) -> float:
    """Gets soil pH from SoilGrids REST API via AOI centroid."""
    try:
        centroid = aoi_geometry.centroid().coordinates().getInfo()
        lon, lat = centroid[0], centroid[1]
        
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&depth=0-5cm&value=mean"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        ph_value = data['properties']['layers'][0]['depths'][0]['values']['mean'] / 10.0
        return ph_value
    except Exception as e:
        print(f"WARNING: Could not fetch soil pH: {e}. Returning default value.")
        return 6.5 # Return a neutral default

# --- Main Pipeline Function ---

def build_site_fingerprint(aoi: AOI) -> SiteFingerprint:
    """
    Generates a comprehensive ecological fingerprint for a given Area of Interest (AOI).
    This is the single, authoritative entry point for Phase 1: Site Assessment.
    """
    if not initialize_gee():
        raise RuntimeError("Could not initialize Google Earth Engine. Pipeline cannot run.")

    print(f"INFO: Building site fingerprint for AOI: {aoi.name}...")
    
    try:
        aoi_geometry = ee.Geometry.Polygon(aoi.polygon_geojson['coordinates'])
    except Exception as e:
        raise ValueError(f"Invalid GeoJSON provided for AOI '{aoi.name}': {e}")

    # --- Execute Data Collection in Parallel (Conceptually) ---
    # In a real-world scenario, you might use concurrent futures for REST APIs
    
    print("INFO: Fetching NDVI, rainfall, and soil data...")
    ndvi_data = _get_sentinel2_ndvi_stats(aoi_geometry)
    rainfall_data = _get_chirps_rainfall(aoi_geometry)
    soil_ph_data = _get_soil_ph(aoi_geometry)
    
    # --- Assemble the Final Fingerprint using the Pydantic Schema ---
    fingerprint = SiteFingerprint(
        aoi_name=aoi.name,
        ndvi_stats={"annual": Stats(**ndvi_data)}, # Simplified to annual for now
        rainfall_mm_year=rainfall_data,
        soil_ph_estimate=soil_ph_data,
        # Slope stats can be added here using ee.Terrain(SRTM_IMAGE).select('slope')
        slope_stats=None # Placeholder for now
    )
    
    print(f"✅ INFO: Successfully built site fingerprint for {aoi.name}.")
    return fingerprint