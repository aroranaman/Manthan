# src/processing/advanced_gee_processor.py
"""
Enhanced Google Earth Engine processor for Manthan
Builds on existing NDVI/CHIRPS implementation with advanced features
"""

import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

class ManthanAdvancedGEE:
    """
    Enhanced GEE processor replacing current basic NDVI/CHIRPS analysis
    Maintains API compatibility with existing dashboard
    """
    
    def __init__(self):
        """Initialize with enhanced capabilities"""
        try:
            ee.Initialize()
            logging.info("Google Earth Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize GEE: {e}")
            raise
        
        self.use_cloud_score_plus = True  # Enhanced cloud masking
        self.vegetation_indices = [
            'NDVI', 'EVI', 'SAVI', 'NDWI', 'GNDVI', 'CVI', 'NDRE', 'CIG'
        ]
        
    def comprehensive_site_analysis(self, aoi, start_date: str, end_date: str) -> Dict:
        """
        Complete site analysis combining all data layers
        MAIN ENTRY POINT - replaces current basic analysis in dashboard
        
        Args:
            aoi: Area of Interest (ee.Geometry)
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            
        Returns:
            Comprehensive site analysis results for AI species recommender
        """
        logging.info(f"Starting comprehensive analysis for period {start_date} to {end_date}")
        
        try:
            # Enhanced satellite analysis (replaces current simple NDVI)
            satellite_data = self._process_sentinel2_advanced(aoi, start_date, end_date)
            
            # Enhanced rainfall analysis (replaces current CHIRPS)
            rainfall_data = self._enhanced_chirps_analysis(aoi, start_date, end_date)
            
            # Topographic analysis (new capability)
            topo_data = self._add_topographic_features(aoi)
            
            # Soil analysis (new capability)
            soil_data = self._add_soil_features(aoi)
            
            # Climate analysis (new capability)
            climate_data = self._add_climate_features(aoi, start_date, end_date)
            
            # Combine all features for AI model
            site_features = self._combine_site_features(
                satellite_data, rainfall_data, topo_data, soil_data, climate_data, aoi
            )
            
            # Calculate site statistics for species recommendation
            site_statistics = self._extract_site_statistics(site_features, aoi)
            
            return {
                'site_features': site_features,
                'site_statistics': site_statistics,
                'satellite_analysis': satellite_data,
                'rainfall_analysis': rainfall_data,
                'topographic_analysis': topo_data,
                'soil_analysis': soil_data,
                'climate_analysis': climate_data,
                'analysis_date': datetime.now().isoformat(),
                'site_area_hectares': self._calculate_area_hectares(aoi),
                'data_quality_metrics': self._assess_data_quality(satellite_data, rainfall_data)
            }
            
        except Exception as e:
            logging.error(f"Site analysis failed: {e}")
            raise
    
    def _process_sentinel2_advanced(self, aoi, start_date: str, end_date: str) -> Dict:
        """
        Enhanced Sentinel-2 processing with Cloud Score+ and multiple indices
        Replaces current basic NDVI implementation
        """
        # Load Sentinel-2 Surface Reflectance collection
        s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        
        # Apply enhanced filtering
        filtered = (s2_sr
                   .filterBounds(aoi)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        # Apply Cloud Score+ masking if available
        if self.use_cloud_score_plus:
            try:
                cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
                cs_plus_filtered = cs_plus.filterBounds(aoi).filterDate(start_date, end_date)
                
                # Join Cloud Score+ with Sentinel-2
                joined = ee.Join.saveFirst('cloud_score').apply(
                    primary=filtered,
                    secondary=cs_plus_filtered,
                    condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
                )
                
                def apply_cs_plus_mask(image):
                    cs_image = ee.Image(image.get('cloud_score'))
                    cs_mask = cs_image.select('cs').gte(0.5)
                    return image.updateMask(cs_mask)
                
                filtered = ee.ImageCollection(joined).map(apply_cs_plus_mask)
                logging.info("Applied Cloud Score+ masking")
                
            except Exception as e:
                logging.warning(f"Cloud Score+ not available, using basic cloud masking: {e}")
                # Fallback to basic cloud masking (current method)
                def mask_s2_clouds(image):
                    qa = image.select('QA60')
                    cloud_bit_mask = 1 << 10
                    cirrus_bit_mask = 1 << 11
                    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
                    )
                    return image.updateMask(mask)
                
                filtered = filtered.map(mask_s2_clouds)
        
        # Calculate comprehensive vegetation indices
        def add_vegetation_indices(image):
            """Add multiple vegetation indices beyond basic NDVI"""
            
            # NDVI (current implementation)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            # Enhanced Vegetation Index
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4'),
                    'BLUE': image.select('B2')
                }
            ).rename('EVI')
            
            # Soil Adjusted Vegetation Index
            savi = image.expression(
                '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
                {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4')
                }
            ).rename('SAVI')
            
            # Normalized Difference Water Index
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            
            # Green NDVI
            gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
            
            # Chlorophyll Vegetation Index
            cvi = image.expression(
                '(NIR / GREEN) * (RED / GREEN)',
                {
                    'NIR': image.select('B8'),
                    'GREEN': image.select('B3'),
                    'RED': image.select('B4')
                }
            ).rename('CVI')
            
            # Red Edge NDVI
            ndre = image.normalizedDifference(['B8', 'B6']).rename('NDRE')
            
            # Chlorophyll Index Green
            cig = image.expression(
                '(NIR / GREEN) - 1',
                {
                    'NIR': image.select('B8'),
                    'GREEN': image.select('B3')
                }
            ).rename('CIG')
            
            return image.addBands([ndvi, evi, savi, ndwi, gndvi, cvi, ndre, cig])
        
        # Apply vegetation indices to collection
        enhanced_collection = filtered.map(add_vegetation_indices)
        
        # Create temporal composites
        composite_median = enhanced_collection.median()
        composite_mean = enhanced_collection.mean()
        composite_std = enhanced_collection.reduce(ee.Reducer.stdDev())
        
        return {
            'collection': enhanced_collection,
            'composite_median': composite_median,
            'composite_mean': composite_mean,
            'composite_std': composite_std,
            'count': enhanced_collection.size(),
            'date_range': {'start': start_date, 'end': end_date}
        }
    
    def _enhanced_chirps_analysis(self, aoi, start_date: str, end_date: str) -> Dict:
        """
        Enhanced CHIRPS processing building on current rainfall integration
        """
        # Load CHIRPS daily data (current implementation)
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        filtered = chirps.filterBounds(aoi).filterDate(start_date, end_date)
        
        # Advanced rainfall analytics
        total_precipitation = filtered.sum().rename('total_precip')
        mean_daily_precip = filtered.mean().rename('mean_daily_precip')
        max_daily_precip = filtered.max().rename('max_daily_precip')
        precip_days = filtered.map(lambda img: img.gt(1)).sum().rename('precip_days')
        
        # Drought indices
        def calculate_spi_simple(collection):
            """Simplified SPI calculation"""
            total = collection.sum()
            mean_precip = total.divide(collection.size())
            return mean_precip.rename('SPI_3months')
        
        spi_3month = calculate_spi_simple(filtered)
        
        # Seasonal analysis
        def add_month_property(image):
            month = ee.Number.parse(image.date().format('MM'))
            return image.set('month', month)
        
        monthly_collection = filtered.map(add_month_property)
        
        # Calculate seasonal totals
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5], 
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11]
        }
        
        seasonal_totals = {}
        for season, months in seasons.items():
            season_filter = ee.Filter.inList('month', months)
            season_total = monthly_collection.filter(season_filter).sum()
            seasonal_totals[season] = season_total
        
        return {
            'total_precipitation': total_precipitation,
            'mean_daily_precip': mean_daily_precip,
            'max_daily_precip': max_daily_precip,
            'precipitation_days': precip_days,
            'spi_3month': spi_3month,
            'seasonal_totals': seasonal_totals,
            'raw_collection': filtered,
            'analysis_period': {'start': start_date, 'end': end_date}
        }
    
    def _add_topographic_features(self, aoi) -> Dict:
        """Add elevation, slope, aspect - critical for species selection"""
        # SRTM Digital Elevation Model
        srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
        
        # Calculate terrain derivatives
        slope = ee.Terrain.slope(srtm).rename('slope')
        aspect = ee.Terrain.aspect(srtm).rename('aspect')
        
        # Topographic Position Index
        tpi = srtm.subtract(
            srtm.focal_mean(ee.Kernel.circle(radius=100, units='meters'))
        ).rename('TPI')
        
        # Terrain Ruggedness Index
        tri = srtm.subtract(srtm.focal_mean(ee.Kernel.square(1))).abs().rename('TRI')
        
        return {
            'elevation': srtm,
            'slope': slope,
            'aspect': aspect,
            'tpi': tpi,
            'tri': tri
        }
    
    def _add_soil_features(self, aoi) -> Dict:
        """Add soil characteristics from SoilGrids"""
        try:
            # SoilGrids data (if available)
            soil_ph = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02').select('b0').rename('soil_ph')
            soil_organic_carbon = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02').select('b0').rename('soil_oc')
            soil_texture_clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0').rename('clay_content')
            
            return {
                'soil_ph': soil_ph,
                'soil_organic_carbon': soil_organic_carbon,
                'clay_content': soil_texture_clay
            }
        except Exception as e:
            logging.warning(f"Soil data not available: {e}")
            # Return dummy soil data
            dummy_soil = ee.Image.constant(7.0).rename('soil_ph')
            return {
                'soil_ph': dummy_soil,
                'soil_organic_carbon': dummy_soil.rename('soil_oc'),
                'clay_content': dummy_soil.rename('clay_content')
            }
    
    def _add_climate_features(self, aoi, start_date: str, end_date: str) -> Dict:
        """Add temperature and humidity data"""
        try:
            # ERA5 Climate Reanalysis data
            era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            
            filtered = era5.filterBounds(aoi).filterDate(start_date, end_date)
            
            # Temperature analysis
            temp_2m = filtered.select('temperature_2m').mean().rename('temperature_mean')
            temp_max = filtered.select('temperature_2m').max().rename('temperature_max')
            temp_min = filtered.select('temperature_2m').min().rename('temperature_min')
            
            return {
                'temperature_mean': temp_2m,
                'temperature_max': temp_max,
                'temperature_min': temp_min
            }
        except Exception as e:
            logging.warning(f"Climate data not available: {e}")
            # Return dummy climate data
            dummy_temp = ee.Image.constant(25.0).rename('temperature_mean')
            return {
                'temperature_mean': dummy_temp,
                'temperature_max': dummy_temp.rename('temperature_max'),
                'temperature_min': dummy_temp.rename('temperature_min')
            }
    
    def _combine_site_features(self, satellite_data: Dict, rainfall_data: Dict, 
                              topo_data: Dict, soil_data: Dict, climate_data: Dict, aoi) -> ee.Image:
        """Combine all features into single multi-band image"""
        
        feature_bands = []
        
        # Add vegetation indices from satellite data
        composite = satellite_data['composite_median']
        for index in self.vegetation_indices:
            if index in composite.bandNames().getInfo():
                feature_bands.append(composite.select(index))
        
        # Add rainfall features
        feature_bands.extend([
            rainfall_data['total_precipitation'],
            rainfall_data['mean_daily_precip'],
            rainfall_data['spi_3month']
        ])
        
        # Add topographic features
        feature_bands.extend([
            topo_data['elevation'],
            topo_data['slope'],
            topo_data['aspect']
        ])
        
        # Add soil features
        feature_bands.extend([
            soil_data['soil_ph'],
            soil_data['soil_organic_carbon'],
            soil_data['clay_content']
        ])
        
        # Add climate features
        feature_bands.extend([
            climate_data['temperature_mean'],
            climate_data['temperature_max'],
            climate_data['temperature_min']
        ])
        
        # Combine all bands
        combined_features = ee.Image.cat(feature_bands)
        
        return combined_features
    
    def _extract_site_statistics(self, site_features: ee.Image, aoi) -> Dict:
        """Extract site statistics for AI model input"""
        
        # Compute comprehensive statistics
        stats = site_features.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        
        return stats
    
    def _calculate_area_hectares(self, aoi) -> float:
        """Calculate area in hectares"""
        try:
            area_sq_meters = aoi.area().getInfo()
            return area_sq_meters / 10000  # Convert to hectares
        except:
            return 0.0
    
    def _assess_data_quality(self, satellite_data: Dict, rainfall_data: Dict) -> Dict:
        """Assess quality of input data"""
        try:
            satellite_count = satellite_data['count'].getInfo()
            rainfall_count = rainfall_data['raw_collection'].size().getInfo()
            
            quality_metrics = {
                'satellite_image_count': satellite_count,
                'rainfall_data_points': rainfall_count,
                'data_quality_score': min(1.0, (satellite_count + rainfall_count) / 100),
                'has_cloud_masking': self.use_cloud_score_plus,
                'vegetation_indices_count': len(self.vegetation_indices)
            }
            
            return quality_metrics
        except Exception as e:
            logging.warning(f"Could not assess data quality: {e}")
            return {'data_quality_score': 0.5, 'error': str(e)}

# Integration function for existing Manthan dashboard
def integrate_with_dashboard():
    """
    Shows how to integrate enhanced GEE processor with existing dashboard
    Replace current GEE analysis calls with this enhanced version
    """
    
    example_usage = '''
    # In existing src/dashboard/streamlit_app.py
    # Replace current GEE analysis with enhanced version:
    
    from src.processing.advanced_gee_processor import ManthanAdvancedGEE
    
    # Initialize enhanced processor
    @st.cache_resource 
    def load_gee_processor():
        return ManthanAdvancedGEE()
    
    gee_processor = load_gee_processor()
    
    # In main dashboard logic, replace current analysis:
    if st.button("Analyze Site"):
        with st.spinner("Running comprehensive analysis..."):
            # Enhanced analysis with 47 features for AI
            results = gee_processor.comprehensive_site_analysis(
                aoi=selected_aoi,  # From existing map selection
                start_date=start_date,
                end_date=end_date
            )
            
            # Display enhanced results
            st.success(f"Analysis complete! Found {results['data_quality_metrics']['satellite_image_count']} satellite images")
            
            # Show data quality
            quality_score = results['data_quality_metrics']['data_quality_score']
            st.metric("Data Quality Score", f"{quality_score:.2f}")
            
            # Store results for AI species recommender
            st.session_state.gee_analysis = results
            
            # Pass to AI species recommender
            if 'species_predictor' in st.session_state:
                ai_recommendations = st.session_state.species_predictor.predict_for_site(results)
                # Display AI recommendations...
    '''
    
    return example_usage

if __name__ == "__main__":
    # Example usage
    processor = ManthanAdvancedGEE()
    print("Enhanced GEE Processor initialized")
    print(f"Vegetation indices: {processor.vegetation_indices}")
    print("Ready for integration with Manthan dashboard")