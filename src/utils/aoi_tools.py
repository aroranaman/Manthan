"""
Area of Interest (AOI) Processing Tools
======================================
Utilities for validating, processing, and analyzing user-selected areas.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
import numpy as np
import ee
from typing import Dict, Tuple, Optional, List

class AOIProcessor:
    """Handles AOI validation and processing."""
    
    def __init__(self):
        """Initialize AOI processor with India boundary."""
        self.india_bounds = self._load_india_boundary()
        
    def _load_india_boundary(self) -> ee.Geometry:
        """Load India administrative boundary from GEE."""
        try:
            india = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
                ee.Filter.eq('ADM0_NAME', 'India')
            )
            return india.geometry()
        except:
            # Fallback bounding box
            return ee.Geometry.Rectangle([68.0, 6.0, 98.0, 38.0])
    
    def create_aoi_from_coords(self, coordinates: List[List[float]]) -> Dict:
        """
        Create AOI from coordinate list.
        
        Args:
            coordinates: List of [lon, lat] coordinate pairs
            
        Returns:
            Dictionary with AOI information
        """
        try:
            # Create Shapely polygon
            polygon = Polygon(coordinates)
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {'id': [1]}, 
                geometry=[polygon], 
                crs='EPSG:4326'
            )
            
            # Calculate area
            area_ha = self.calculate_area(gdf)
            centroid = self.get_centroid(gdf)
            
            # Validate AOI
            is_valid, message = self.validate_aoi(polygon, area_ha)
            
            # Create Earth Engine geometry
            ee_geometry = ee.Geometry.Polygon(coordinates)
            
            return {
                'gdf': gdf,
                'polygon': polygon,
                'ee_geometry': ee_geometry,
                'area_ha': area_ha,
                'centroid': centroid,
                'is_valid': is_valid,
                'validation_message': message,
                'bounds': list(polygon.bounds)  # [minx, miny, maxx, maxy]
            }
            
        except Exception as e:
            return {
                'error': f"AOI creation failed: {str(e)}",
                'is_valid': False
            }
    
    def calculate_area(self, gdf: gpd.GeoDataFrame) -> float:
        """Calculate area in hectares."""
        # Project to equal-area projection for India (Albers)
        gdf_projected = gdf.to_crs('EPSG:7755')  # India-specific Albers
        area_m2 = gdf_projected.geometry.area.iloc[0]
        return area_m2 / 10000  # Convert to hectares
    
    def get_centroid(self, gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
        """Get centroid coordinates (lon, lat)."""
        gdf_proj = gdf.to_crs("EPSG:7755")
        centroid_proj = gdf_proj.geometry.centroid.iloc[0]
        # Project centroid back to WGS-84
        centroid = gpd.GeoSeries([centroid_proj], 
    crs="EPSG:7755").to_crs("EPSG:4326").iloc[0]
        return centroid.x, centroid.y
    
    def validate_aoi(self, polygon: Polygon, area_ha: float) -> Tuple[bool, str]:
        """
        Validate AOI against various criteria.
        
        Args:
            polygon: Shapely polygon
            area_ha: Area in hectares
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if polygon is valid
        if not polygon.is_valid:
            return False, "Invalid polygon geometry"
        
        # Check area constraints
        if area_ha < 1:
            return False, f"Area too small ({area_ha:.1f} ha). Minimum: 1 hectare"
        
        if area_ha > 50000:
            return False, f"Area too large ({area_ha:.1f} ha). Maximum: 50,000 hectares"
        
        # Check if within India (simplified bounding box check)
        bounds = polygon.bounds
        india_bbox = [68.0, 6.0, 98.0, 38.0]  # [minx, miny, maxx, maxy]
        
        if not (india_bbox[0] <= bounds[0] and bounds[2] <= india_bbox[2] and
                india_bbox[1] <= bounds[1] and bounds[3] <= india_bbox[3]):
            return False, "AOI appears to be outside India boundaries"
        
        return True, f"Valid AOI: {area_ha:,.1f} hectares"
    
    def get_aoi_info(self, aoi_data: Dict) -> Dict:
        """Get comprehensive AOI information."""
        if 'error' in aoi_data:
            return aoi_data
            
        return {
            'area_hectares': aoi_data['area_ha'],
            'area_km2': aoi_data['area_ha'] / 100,
            'centroid_lon': aoi_data['centroid'][0],
            'centroid_lat': aoi_data['centroid'][1],
            'bounds': aoi_data['bounds'],
            'is_valid': aoi_data['is_valid'],
            'message': aoi_data['validation_message']
        }

def create_sample_aoi() -> Dict:
    """Create a sample AOI for testing (near Pune, Maharashtra)."""
    # Small area near Pune
    coords = [
        [73.8567, 18.5204],  # Southwest
        [73.8667, 18.5204],  # Southeast  
        [73.8667, 18.5304],  # Northeast
        [73.8567, 18.5304],  # Northwest
        [73.8567, 18.5204]   # Close polygon
    ]
    
    processor = AOIProcessor()
    return processor.create_aoi_from_coords(coords)

# Test function
if __name__ == "__main__":
    import ee
    
    try:
        ee.Initialize(project='manthan-466509')
        
        # Test with sample AOI
        aoi_data = create_sample_aoi()
        processor = AOIProcessor()
        info = processor.get_aoi_info(aoi_data)
        
        print("🗺️ AOI Test Results:")
        print(f"Area: {info['area_hectares']:.1f} hectares ({info['area_km2']:.2f} km²)")
        print(f"Centroid: {info['centroid_lat']:.4f}, {info['centroid_lon']:.4f}")
        print(f"Valid: {info['is_valid']} - {info['message']}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
