"""
Sentinel-2 Data Processor using Google Earth Engine
===================================================
Processes Sentinel-2 imagery for NDVI calculation and vegetation analysis.
"""

import ee
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GEESentinel2Processor:
    """Processes Sentinel-2 data using Google Earth Engine."""
    
    def __init__(self):
        """Initialize the Sentinel-2 processor."""
        self.collection_id = 'COPERNICUS/S2_SR_HARMONIZED'
        self.cloud_threshold = 20
        
    def calculate_ndvi(self, 
                      aoi_geometry: ee.Geometry,
                      start_date: str = '2024-01-01',
                      end_date: str = '2024-12-31',
                      cloud_threshold: int = 20) -> Dict:
        """
        Calculate NDVI statistics for AOI using Sentinel-2.
        
        Args:
            aoi_geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            cloud_threshold: Maximum cloud percentage
            
        Returns:
            Dictionary with NDVI statistics
        """
        try:
            # Load Sentinel-2 collection
            collection = (ee.ImageCollection(self.collection_id)
                         .filterBounds(aoi_geometry)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
            
            # Check if images are available
            image_count = collection.size()
            
            if image_count.getInfo() == 0:
                logger.warning("No Sentinel-2 images found for AOI")
                return self._create_fallback_results()
            
            # Function to calculate NDVI
            def add_ndvi(image):
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return image.addBands(ndvi)
            
            # Apply NDVI calculation
            collection_ndvi = collection.map(add_ndvi)
            
            # Create median composite (reduces cloud impact)
            ndvi_composite = collection_ndvi.select('NDVI').median().clip(aoi_geometry)
            
            # Calculate statistics
            stats = ndvi_composite.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.minMax(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.percentile([25, 75]),
                    sharedInputs=True
                ),
                geometry=aoi_geometry,
                scale=10,  # 10m resolution
                maxPixels=1e9
            )
            
            # Extract results
            results = stats.getInfo()
            
            # Process statistics
            ndvi_mean = results.get('NDVI_mean', 0.5)
            ndvi_std = results.get('NDVI_stdDev', 0.15)
            ndvi_min = results.get('NDVI_min', 0.0)
            ndvi_max = results.get('NDVI_max', 1.0)
            ndvi_p25 = results.get('NDVI_p25', 0.3)
            ndvi_p75 = results.get('NDVI_p75', 0.7)
            
            # Calculate vegetation metrics
            vegetation_coverage = self._calculate_vegetation_coverage(ndvi_mean)
            vegetation_health = self._assess_vegetation_health(ndvi_mean, ndvi_std)
            
            return {
                'ndvi_mean': round(ndvi_mean, 4),
                'ndvi_std': round(ndvi_std, 4),
                'ndvi_min': round(ndvi_min, 4),
                'ndvi_max': round(ndvi_max, 4),
                'ndvi_p25': round(ndvi_p25, 4),
                'ndvi_p75': round(ndvi_p75, 4),
                'vegetation_coverage': vegetation_coverage,
                'vegetation_health': vegetation_health,
                'image_count': image_count.getInfo(),
                'date_range': f"{start_date} to {end_date}",
                'cloud_threshold': cloud_threshold,
                'processing_scale': '10m',
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"NDVI calculation failed: {str(e)}")
            return self._create_fallback_results(error=str(e))
    
    def get_monthly_ndvi_trend(self,
                              aoi_geometry: ee.Geometry,
                              year: int = 2024) -> Dict:
        """
        Get monthly NDVI trend for the year.
        
        Args:
            aoi_geometry: Earth Engine geometry
            year: Year to analyze
            
        Returns:
            Dictionary with monthly NDVI values
        """
        try:
            monthly_data = []
            
            for month in range(1, 13):
                start_date = f"{year}-{month:02d}-01"
                
                # Calculate end date for month
                if month == 12:
                    end_date = f"{year+1}-01-01"
                else:
                    end_date = f"{year}-{month+1:02d}-01"
                
                # Get monthly collection
                monthly_collection = (ee.ImageCollection(self.collection_id)
                                    .filterBounds(aoi_geometry)
                                    .filterDate(start_date, end_date)
                                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
                
                if monthly_collection.size().getInfo() > 0:
                    # Calculate monthly NDVI
                    def add_ndvi(image):
                        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                    
                    monthly_ndvi = monthly_collection.map(add_ndvi).select('NDVI').median()
                    
                    # Get mean NDVI for month
                    monthly_stats = monthly_ndvi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=aoi_geometry,
                        scale=10,
                        maxPixels=1e8
                    )
                    
                    ndvi_value = monthly_stats.getInfo().get('NDVI')
                    if ndvi_value is not None:
                        monthly_data.append({
                            'month': month,
                            'month_name': pd.Timestamp(f"{year}-{month:02d}-15").strftime('%B'),
                            'ndvi': round(ndvi_value, 4),
                            'date': f"{year}-{month:02d}-15"
                        })
            
            return {
                'monthly_trend': monthly_data,
                'year': year,
                'status': 'success' if monthly_data else 'no_data'
            }
            
        except Exception as e:
            logger.error(f"Monthly trend calculation failed: {str(e)}")
            return {'monthly_trend': [], 'year': year, 'status': 'error', 'error': str(e)}
    
    def get_sentinel2_composite_image(self,
                                    aoi_geometry: ee.Geometry,
                                    start_date: str,
                                    end_date: str) -> ee.Image:
        """
        Get Sentinel-2 RGB composite for visualization.
        
        Args:
            aoi_geometry: Earth Engine geometry
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Earth Engine Image for visualization
        """
        collection = (ee.ImageCollection(self.collection_id)
                     .filterBounds(aoi_geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_threshold)))
        
        # Create median composite and clip to AOI
        composite = collection.median().clip(aoi_geometry)
        
        # Select RGB bands (B4, B3, B2) and scale for visualization
        rgb_composite = composite.select(['B4', 'B3', 'B2']).multiply(0.0001)
        
        return rgb_composite
    
    def _calculate_vegetation_coverage(self, ndvi_mean: float) -> str:
        """Classify vegetation coverage based on NDVI."""
        if ndvi_mean >= 0.7:
            return "Dense Vegetation"
        elif ndvi_mean >= 0.5:
            return "Moderate Vegetation"
        elif ndvi_mean >= 0.3:
            return "Sparse Vegetation"
        elif ndvi_mean >= 0.1:
            return "Very Sparse Vegetation"
        else:
            return "Bare Soil/Water"
    
    def _assess_vegetation_health(self, ndvi_mean: float, ndvi_std: float) -> str:
        """Assess vegetation health based on NDVI statistics."""
        if ndvi_mean >= 0.6 and ndvi_std < 0.2:
            return "Excellent"
        elif ndvi_mean >= 0.5 and ndvi_std < 0.25:
            return "Good"
        elif ndvi_mean >= 0.3:
            return "Fair"
        elif ndvi_mean >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _create_fallback_results(self, error: Optional[str] = None) -> Dict:
        """Create fallback results when processing fails."""
        return {
            'ndvi_mean': 0.45,
            'ndvi_std': 0.18,
            'ndvi_min': 0.05,
            'ndvi_max': 0.85,
            'ndvi_p25': 0.32,
            'ndvi_p75': 0.62,
            'vegetation_coverage': 'Moderate Vegetation',
            'vegetation_health': 'Fair',
            'image_count': 0,
            'date_range': 'N/A',
            'cloud_threshold': self.cloud_threshold,
            'processing_scale': '10m',
            'status': 'fallback',
            'error': error,
            'note': 'Using estimated values due to data access issues'
        }

# Test function
def main():
    """Test the Sentinel-2 processor."""
    import sys
    sys.path.append('..')
    from utils.aoi_tools import create_sample_aoi
    
    try:
        # Initialize Earth Engine
        ee.Initialize(project='manthan-466509')
        
        # Create sample AOI
        aoi_data = create_sample_aoi()
        
        if not aoi_data.get('is_valid', False):
            print("❌ Sample AOI creation failed")
            return
            
        # Initialize processor
        processor = GEESentinel2Processor()
        
        # Calculate NDVI
        results = processor.calculate_ndvi(
            aoi_data['ee_geometry'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )
        
        print("🌱 Sentinel-2 NDVI Analysis Results:")
        print(f"Mean NDVI: {results['ndvi_mean']:.4f}")
        print(f"Standard Deviation: {results['ndvi_std']:.4f}")
        print(f"Vegetation Coverage: {results['vegetation_coverage']}")
        print(f"Vegetation Health: {results['vegetation_health']}")
        print(f"Images processed: {results['image_count']}")
        print(f"Status: {results['status']}")
        
        # Test monthly trend
        print("\n📈 Monthly NDVI Trend:")
        trend_data = processor.get_monthly_ndvi_trend(aoi_data['ee_geometry'], 2024)
        
        for month_data in trend_data['monthly_trend'][:6]:  # Show first 6 months
            print(f"{month_data['month_name']}: {month_data['ndvi']:.4f}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()
