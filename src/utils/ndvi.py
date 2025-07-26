# src/utils/ndvi.py
import ee
import numpy as np
from utils.gee_auth import gee_init
gee_init() # Initialize Earth Engine


def compute_sentinel2_ndvi(aoi_geometry, start_date='2024-01-01', end_date='2024-12-31'):
    """
    Compute NDVI from Sentinel-2 data for given AOI.
    
    Args:
        aoi_geometry: Earth Engine Geometry object
        start_date: Start date for image collection
        end_date: End date for image collection
    
    Returns:
        dict: NDVI statistics and image
    """
    gee_init()  # ✅ FIXED - removed the extra colon

    try:  # Reflectance collection
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(aoi_geometry)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .select(['B4', 'B8']))  # Red and NIR bands
        
        # Calculate NDVI for each image
        def calculate_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        # Apply NDVI calculation to collection
        ndvi_collection = s2_collection.map(calculate_ndvi)
        
        # Get median NDVI for the time period
        ndvi_median = ndvi_collection.select('NDVI').median().clip(aoi_geometry)
        
        # Calculate statistics
        stats = ndvi_median.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ),
            geometry=aoi_geometry,
            scale=10,  # 10m resolution
            maxPixels=1e9
        )
        
        ndvi_stats = stats.getInfo()
        
        # Interpret vegetation coverage
        mean_ndvi = ndvi_stats.get('NDVI_mean', 0)
        if mean_ndvi > 0.7:
            coverage = "Dense Vegetation"
        elif mean_ndvi > 0.4:
            coverage = "Moderate Vegetation"
        elif mean_ndvi > 0.2:
            coverage = "Sparse Vegetation"
        else:
            coverage = "Bare/Water"
        
        return {
            'ndvi_image': ndvi_median,
            'ndvi_mean': mean_ndvi,
            'ndvi_std': ndvi_stats.get('NDVI_stdDev', 0),
            'ndvi_min': ndvi_stats.get('NDVI_min', 0),
            'ndvi_max': ndvi_stats.get('NDVI_max', 0),
            'vegetation_coverage': coverage,
            'image_count': ndvi_collection.size().getInfo()
        }
        
    except Exception as e:
        print(f"Error computing NDVI: {e}")
        return {
            'ndvi_image': None,
            'ndvi_mean': 0.3,
            'ndvi_std': 0.1,
            'ndvi_min': 0.1,
            'ndvi_max': 0.5,
            'vegetation_coverage': "Data Unavailable",
            'image_count': 0
        }
