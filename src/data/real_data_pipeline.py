# FILE: src/data/real_data_pipeline.py
# FINAL VERSION: Enhanced with a fallback API for soil data and new socioeconomic context.

import requests
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import aiohttp
import asyncio
from dataclasses import dataclass, asdict
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Structures ---
@dataclass
class RealEnvironmentalData:
    """Standardized structure for real environmental data from multiple sources."""
    latitude: float
    longitude: float
    
    # Climate Data
    annual_precipitation: float  # mm
    mean_temperature: float      # °C
    
    # Elevation Data
    elevation: float             # meters
    
    # Soil Data
    soil_ph: float               # pH units
    soil_organic_carbon: float   # g/kg
    clay_content: float          # %
    sand_content: float          # %
    
    # Vegetation Data
    ndvi_mean: float             # NDVI
    
    # Forest Data (from ISFR estimates)
    forest_cover_2021: float     # %
    canopy_density: str          # Dense/Moderate/Open
    
    # --- NEW: User-Centric Socioeconomic Data ---
    agro_ecological_zone: str
    unique_challenges: str
    
    # Metadata
    data_timestamp: str
    confidence_score: float

# --- Data Provider Classes ---

class WorldClimDataProvider:
    """Fetches climate data using robust, multi-API fallbacks."""
    
    async def get_climate_data(self, lat: float, lon: float, session: aiohttp.ClientSession) -> Dict:
        """Primary method to get climate data, trying multiple sources."""
        try:
            data = await self._fetch_open_meteo(lat, lon, session)
            if data: return data
        except Exception as e:
            logger.warning(f"Open-Meteo failed: {e}")
        
        logger.warning("Falling back to estimated climate data.")
        return self._estimate_climate_data(lat)

    async def _fetch_open_meteo(self, lat: float, lon: float, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetches historical climate normals from Open-Meteo."""
        url = "https://climate-api.open-meteo.com/v1/climate"
        params = {
            'latitude': lat, 'longitude': lon, 'start_date': '1991-01-01',
            'end_date': '2020-12-31', 'models': 'FGOALS_f3_H',
            'daily': ['temperature_2m_mean', 'precipitation_sum']
        }
        async with session.get(url, params=params, timeout=20) as response:
            if response.status == 200:
                data = await response.json()
                daily_data = data.get('daily', {})
                temps = daily_data.get('temperature_2m_mean', [])
                precips = daily_data.get('precipitation_sum', [])
                if temps and precips:
                    return {
                        'annual_precipitation': np.sum(precips),
                        'mean_temperature': np.mean(temps),
                        'source': 'Open-Meteo'
                    }
        return None

    def _estimate_climate_data(self, lat: float) -> Dict:
        """Fallback climate estimation based on latitude."""
        temp = 15 + (25 - abs(lat)) * 0.8
        precip = max(200, 1500 - abs(lat) * 30)
        return {
            'annual_precipitation': max(100, precip),
            'mean_temperature': temp,
            'source': 'estimated'
        }

class ElevationProvider:
    """Fetches elevation data from the Open-Elevation API."""
    
    async def get_elevation_data(self, lat: float, lon: float, session: aiohttp.ClientSession) -> Dict:
        """Primary method to get elevation data."""
        url = "https://api.open-elevation.com/api/v1/lookup"
        data = {'locations': [{'latitude': lat, 'longitude': lon}]}
        try:
            async with session.post(url, json=data, timeout=15) as response:
                if response.status == 200:
                    result = await response.json()
                    elevation = result['results'][0]['elevation']
                    return {'elevation': float(elevation), 'source': 'Open-Elevation'}
        except Exception as e:
            logger.warning(f"Open-Elevation failed: {e}")
        
        logger.warning("Falling back to estimated elevation data.")
        return {'elevation': 500.0, 'source': 'estimated'}

class SoilDataProvider:
    """Fetches soil data from multiple sources for resilience."""
    
    async def get_soil_data(self, lat: float, lon: float, session: aiohttp.ClientSession) -> Dict:
        """Get soil data, trying SoilGrids first, then OpenLandMap as a fallback."""
        try:
            soil_data = await self._fetch_soilgrids_api(lat, lon, session)
            if soil_data: return soil_data
        except Exception as e:
            logger.warning(f"SoilGrids failed: {e}. Trying fallback.")

        try:
            soil_data = await self._fetch_openlandmap_api(lat, lon, session)
            if soil_data: return soil_data
        except Exception as e:
            logger.warning(f"OpenLandMap also failed: {e}. Resorting to estimation.")

        return self._estimate_soil_data()

    async def _fetch_soilgrids_api(self, lat: float, lon: float, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetches from SoilGrids REST API."""
        base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {'lon': lon, 'lat': lat, 'property': ['phh2o', 'ocd', 'clay', 'sand'], 'depth': ['0-5cm'], 'value': ['mean']}
        async with session.get(base_url, params=params, timeout=20) as response:
            if response.status == 200:
                data = await response.json()
                props = {layer['name']: layer['depths'][0]['values']['mean'] for layer in data['properties']['layers']}
                return {'soil_ph': props.get('phh2o', 70) / 10.0, 'soil_organic_carbon': props.get('ocd', 100) / 10.0, 'clay_content': props.get('clay', 250) / 10.0, 'sand_content': props.get('sand', 450) / 10.0, 'source': 'SoilGrids'}
        return None

    async def _fetch_openlandmap_api(self, lat: float, lon: float, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetches from OpenLandMap REST API as a fallback."""
        base_url = "https://api.openlandmap.org/query/point"
        layers = "phh2o_0-5cm_mean,soc_0-5cm_mean,clay_0-5cm_mean,sand_0-5cm_mean"
        url = f"{base_url}?lon={lon}&lat={lat}&layers={layers}"
        async with session.get(url, timeout=20) as response:
            if response.status == 200:
                data = await response.json()
                props = data.get('layers', {})
                return {'soil_ph': props.get('phh2o_0-5cm_mean', 70) / 10.0, 'soil_organic_carbon': props.get('soc_0-5cm_mean', 100) / 10.0, 'clay_content': props.get('clay_0-5cm_mean', 250) / 10.0, 'sand_content': props.get('sand_0-5cm_mean', 450) / 10.0, 'source': 'OpenLandMap'}
        return None

    def _estimate_soil_data(self) -> Dict:
        """Fallback soil estimation if all APIs fail."""
        return {'soil_ph': 7.0, 'soil_organic_carbon': 15.0, 'clay_content': 30.0, 'sand_content': 40.0, 'source': 'estimated'}

class SocioEconomicProvider:
    """
    Provides user-centric demographic and regional context.
    Uses a rule-based system based on location to identify major zones and their challenges.
    """
    def get_socio_economic_data(self, lat: float, lon: float) -> Dict:
        """Determines the agro-ecological zone and its unique problems."""
        # Simplified boundaries for major Indian regions
        if 24 <= lat <= 31 and 75 <= lon <= 89:
            return {
                'agro_ecological_zone': 'Indo-Gangetic Plains',
                'unique_challenges': 'High population density, intensive agriculture, water table depletion, soil nutrient loss.',
                'source': 'Rule-Based Classification'
            }
        elif 8 <= lat <= 13 and 74 <= lon <= 77:
            return {
                'agro_ecological_zone': 'Western Ghats',
                'unique_challenges': 'High biodiversity, steep slopes prone to erosion, intense monsoon rainfall, human-wildlife conflict.',
                'source': 'Rule-Based Classification'
            }
        elif 24 <= lat <= 32 and 69 <= lon <= 76:
            return {
                'agro_ecological_zone': 'Thar Desert & Arid Plains',
                'unique_challenges': 'Extreme water scarcity, high temperatures, soil salinity, desertification risk.',
                'source': 'Rule-Based Classification'
            }
        elif 12 <= lat <= 25 and 74 <= lon <= 85:
            return {
                'agro_ecological_zone': 'Deccan Plateau',
                'unique_challenges': 'Semi-arid conditions, reliance on rain-fed agriculture, land degradation, water stress.',
                'source': 'Rule-Based Classification'
            }
        else:
            return {
                'agro_ecological_zone': 'Mixed Zone',
                'unique_challenges': 'Variable conditions, requires highly localized analysis.',
                'source': 'Rule-Based Classification'
            }

# --- Main Data Aggregator ---
class RealDataAggregator:
    """Aggregates all real environmental and socioeconomic data sources asynchronously."""
    
    def __init__(self):
        self.worldclim = WorldClimDataProvider()
        self.elevation = ElevationProvider()
        self.soil = SoilDataProvider()
        self.socioeconomic = SocioEconomicProvider() # NEW
        self.cache = {}
        self.cache_duration = timedelta(hours=24)
    
    async def get_complete_environmental_data(self, lat: float, lon: float) -> RealEnvironmentalData:
        """Fetches and combines data from all sources for a given location."""
        cache_key = f"{lat:.4f},{lon:.4f}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.info(f"Using cached data for {lat:.4f}, {lon:.4f}")
                return cached_data
        
        logger.info(f"Fetching new environmental data for {lat:.4f}, {lon:.4f}")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.worldclim.get_climate_data(lat, lon, session),
                self.elevation.get_elevation_data(lat, lon, session),
                self.soil.get_soil_data(lat, lon, session)
            ]
            climate, elevation, soil = await asyncio.gather(*tasks)
            
        # Synchronous call for socioeconomic data
        socioeconomic = self.socioeconomic.get_socio_economic_data(lat, lon)
            
        # Placeholders for MODIS and ISFR data
        vegetation = {'ndvi_mean': 0.5, 'source': 'estimated'}
        forest = {'forest_cover_2021': 25.0, 'canopy_density': 'Moderate', 'source': 'estimated'}
            
        sources = [climate['source'], elevation['source'], soil['source'], vegetation['source'], forest['source'], socioeconomic['source']]
        confidence = np.mean([0.9 if s not in ['estimated', 'unknown', 'Rule-Based Classification'] else 0.4 for s in sources])
            
        env_data = RealEnvironmentalData(
            latitude=lat, longitude=lon,
            annual_precipitation=climate['annual_precipitation'],
            mean_temperature=climate['mean_temperature'],
            elevation=elevation['elevation'],
            soil_ph=soil['soil_ph'],
            soil_organic_carbon=soil['soil_organic_carbon'],
            clay_content=soil['clay_content'],
            sand_content=soil['sand_content'],
            ndvi_mean=vegetation['ndvi_mean'],
            forest_cover_2021=forest['forest_cover_2021'],
            canopy_density=forest['canopy_density'],
            agro_ecological_zone=socioeconomic['agro_ecological_zone'],
            unique_challenges=socioeconomic['unique_challenges'],
            data_timestamp=datetime.now().isoformat(),
            confidence_score=confidence
        )
            
        self.cache[cache_key] = (env_data, datetime.now())
        return env_data

async def get_real_environmental_data(lat: float, lon: float) -> Dict:
    """Main function to be called by other modules to get environmental data."""
    aggregator = RealDataAggregator()
    env_data = await aggregator.get_complete_environmental_data(lat, lon)
    return asdict(env_data)

# --- Testing and Validation ---
async def test_data_pipeline():
    """Tests the complete data pipeline with several diverse locations in India."""
    test_locations = [
        (28.61, 77.23, "Delhi (Indo-Gangetic Plains)"),
        (10.85, 76.27, "Kerala (Western Ghats)"),
        (26.91, 75.79, "Jaipur (Arid Plains)"),
        (17.38, 78.48, "Hyderabad (Deccan Plateau)")
    ]
    
    print("🧪 Testing Resilient & Context-Aware Data Pipeline")
    print("=" * 60)
    
    for lat, lon, location in test_locations:
        print(f"\n📍 Testing: {location} ({lat:.2f}, {lon:.2f})")
        try:
            start_time = datetime.now()
            env_data = await get_real_environmental_data(lat, lon)
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Success in {duration:.2f}s")
            print(f"   - 🌍 Agro-Ecological Zone: {env_data['agro_ecological_zone']}")
            print(f"   - ⚠️ Unique Challenges: {env_data['unique_challenges']}")
            print(f"   - 📊 Confidence: {env_data['confidence_score']:.2f}")
            
        except Exception as e:
            print(f"❌ Failed for {location}: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_pipeline())
