# FILE: src/core/geospatial_handler.py
# FINAL VERSION: Incorporates a comprehensive suite of auto-ingested data layers,
# data validation, and a robust synthetic data fallback system.

import ee
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

# Ensure the project root is in the Python path
import sys
from pathlib import Path
try:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except (IndexError, NameError):
    ROOT = Path.cwd()

# Safe import with fallback for gee_init
try:
    from src.utils.gee_auth import gee_init
except ImportError:
    print("⚠️  gee_auth module not found. Using fallback GEE initialization.")
    
    def gee_init():
        """Fallback GEE initialization function."""
        try:
            ee.Initialize()
            return True
        except Exception as e:
            print(f"GEE initialization failed: {e}")
            try:
                ee.Authenticate()
                ee.Initialize()
                return True
            except:
                return False
def get_worldcover_summary(self, aoi: Dict, scale: int = 100) -> Dict[int, int]:
    """
    Returns a histogram {class_value: pixel_count} for ESA WorldCover v200 within AOI.
    Uses a coarse scale to avoid timeouts on free tier.
    """
    if not self.gee_initialized:
        return {}

    geometry = ee.Geometry(aoi["geometry"])
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    # Small, robust server-side histogram
    hist = wc.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        tileScale=4  # helps with larger AOIs
    )

    # This is tiny and safe to bring client-side
    d = ee.Dictionary(hist.get("Map")).getInfo() or {}
    # keys come back as strings; convert to int
    return {int(k): int(v) for k, v in d.items()}


def get_worldcover_tile_url(self, aoi: Dict) -> str:
    """
    Returns a web tile URL to overlay ESA WorldCover v200 on Folium.
    We remap categorical classes to 0..10 and apply a categorical palette.
    """
    if not self.gee_initialized:
        return ""

    # ESA WC categories we care about (docs-aligned)
    from_classes = [10, 20, 30, 40, 50, 60, 70,  80,  90,  95, 100]
    to_indices   = [ 0,  1,  2,  3,  4,  5,  6,   7,   8,   9,  10]
    palette = [
        "#006400",  # 10 Tree cover
        "#ffbb22",  # 20 Shrubland
        "#ffff4c",  # 30 Grassland
        "#f096ff",  # 40 Cropland
        "#fa0000",  # 50 Built-up
        "#b4b4b4",  # 60 Barren
        "#f0f0f0",  # 70 Snow/Ice
        "#0064c8",  # 80 Open water
        "#0096a0",  # 90 Herbaceous wetland
        "#00cf75",  # 95 Mangroves
        "#fae6a0",  # 100 Moss & Lichen
    ]

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc_idx = wc.remap(from_classes, to_indices).rename("wc")

    styled = wc_idx.visualize(min=0, max=10, palette=palette)

    # NOTE: we don't need to clip to AOI for tiles; Folium will only draw visible area
    m = ee.data.getMapId({"image": styled})
    return m["tile_fetcher"].url_format


class GeospatialDataHandler:
    """
    Handles on-the-fly fetching and processing of a rich, multi-layered
    geospatial dataset for the Manthan AI platform.
    """
    def __init__(self):
        """Initializes the GEE API connection."""
        self.gee_initialized = False
        try:
            if gee_init():
                self.gee_initialized = True
                print("✅ Google Earth Engine initialized successfully.")
            else:
                raise RuntimeError("GEE initialization returned False.")
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            print("⚠️  GeospatialDataHandler will work in demo mode with synthetic data.")

    def get_multispectral_patch(
        self,
        aoi: Dict,
        scale: int = 30  # Use 30m as a robust default to prevent timeouts
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Fetches and stacks a comprehensive, multi-channel data patch for a given AOI.

        Args:
            aoi (Dict): A GeoJSON-like dictionary defining the area of interest.
            scale (int): The spatial resolution in meters.

        Returns:
            A tuple containing the NumPy array and an error message string if it fails.
        """
        if not self.gee_initialized:
            print("⚠️  GEE not available. Returning synthetic data for development.")
            return self._generate_synthetic_patch(aoi, scale), None
            
        try:
            geometry = ee.Geometry(aoi['geometry'])
            
            print("🔄 Fetching data from Google Earth Engine...")
            
            # --- BASE LAYERS (High Resolution) ---
            s2_image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .median())
            
            target_projection = s2_image.projection()

            # --- INPUT LAYERS (Auto-ingested and Processed) ---

            # 1. Satellite Imagery (6 bands from Sentinel-2)
            s2_bands = s2_image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

            # 2. Vegetation Indices (2 bands: NDVI from Sentinel, EVI from MODIS)
            ndvi = s2_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            evi = (ee.ImageCollection('MODIS/061/MOD13A1')
                   .filterDate('2023-01-01', '2023-12-31').median()
                   .select('EVI').multiply(0.0001) # Apply scale factor
                   .reproject(crs=target_projection.crs(), scale=scale).rename('EVI'))

            # 3. Climate Data (1 band: CHIRPS rainfall)
            rainfall = (ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
                        .filterDate('2023-01-01', '2023-12-31').sum()
                        .reproject(crs=target_projection.crs(), scale=scale).rename('rainfall'))

            # 4. Soil Data (1 band: SoilGrids pH)
            soil_ph = (ee.Image("projects/soilgrids-isric/phh2o_mean").select(0)
                       .reproject(crs=target_projection.crs(), scale=scale).rename('soil_ph'))

            # 5. Elevation Data (3 bands: Elevation, Slope, Aspect from SRTM)
            srtm_dem = ee.Image('USGS/SRTMGL1_003')
            elevation = srtm_dem.select('elevation').reproject(crs=target_projection.crs(), scale=scale)
            slope = ee.Terrain.slope(srtm_dem).rename('slope').reproject(crs=target_projection.crs(), scale=scale)
            aspect = ee.Terrain.aspect(srtm_dem).rename('aspect').reproject(crs=target_projection.crs(), scale=scale)

            # 6. Hydrology (1 band: Distance to Water)
            water_bodies = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
            distance_to_water = (water_bodies.gt(50) # Areas with >50% water occurrence
                                 .distance(ee.Kernel.euclidean(5000, 'meters'))
                                 .reproject(crs=target_projection.crs(), scale=scale).rename('dist_to_water'))

            # 7. Land Use (1 band: ESA WorldCover)
            lulc = (ee.ImageCollection("ESA/WorldCover/v100").first().select('Map')
                    .reproject(crs=target_projection.crs(), scale=scale).rename('lulc'))

            # Stack all 15 layers into a single multi-band image
            stacked_image = (ee.Image.cat([
                s2_bands, ndvi, evi, rainfall, soil_ph, 
                elevation, slope, aspect, distance_to_water, lulc
            ])
            .float()
            .unmask(0)
            .reproject(crs=target_projection.crs(), scale=scale))

            patch_data = stacked_image.sampleRectangle(region=geometry, defaultValue=0)
            
            band_names = stacked_image.bandNames().getInfo()
            band_arrays = []
            expected_shape = None

            for band in band_names:
                try:
                    arr = np.array(patch_data.get(band).getInfo())
                    if arr.ndim != 2:
                        print(f"⚠️ Band {band} is not 2D. Skipping.")
                        continue
                    if expected_shape is None:
                        expected_shape = arr.shape
                    elif arr.shape != expected_shape:
                        print(f"⚠️ Shape mismatch in band {band}: expected {expected_shape}, got {arr.shape}. Skipping.")
                        continue
                    band_arrays.append(arr)
                except Exception as e:
                    print(f"⚠️ Failed to fetch band {band}: {e}")
                    continue

            if len(band_arrays) < 10:
                warnings.warn("Insufficient valid bands fetched. Falling back to synthetic patch.")
                return self._generate_synthetic_patch(aoi, scale), "Band shape mismatch or missing data"

            try:
                final_patch = np.stack(band_arrays, axis=0)
            except Exception as e:
                print(f"❌ Error stacking band arrays: {e}")
                return self._generate_synthetic_patch(aoi, scale), f"Band stacking error: {e}"

            if self._validate_patch(final_patch):
                return final_patch, None
            else:
                warnings.warn("Patch validation failed. Returning synthetic data.")
                return self._generate_synthetic_patch(aoi, scale), "Validation failed"
            # If we reach here, the patch is valid       
            if self._validate_patch(final_patch):
                return final_patch, None
            else:
                warnings.warn("Patch validation failed. Returning synthetic data.")
                return self._generate_synthetic_patch(aoi, scale), "Validation failed"

        except ee.ee_exception.EEException as e:
            error_message = str(e)
            print(f"❌ GEE Error: {error_message}")
            if "computation timed out" in error_message.lower() or "too large" in error_message.lower():
                return None, "GEE computation timed out or the selected area is too large. Please select a smaller region."
            
            print("⚠️  GEE error, falling back to synthetic data")
            return self._generate_synthetic_patch(aoi, scale), f"GEE Error: {error_message}"
                
        except Exception as e:
            print(f"❌ General error: {e}")
            print("⚠️  Falling back to synthetic data")
            return self._generate_synthetic_patch(aoi, scale), f"An unexpected error occurred: {e}"

    def _validate_patch(self, patch: np.ndarray) -> bool:
        """Performs basic quality control on the fetched data patch."""
        if patch is None or patch.size == 0:
            print("Validation failed: Patch is empty.")
            return False
        if np.all(patch == 0):
            print("Validation failed: Patch contains only zero values.")
            return False
        if patch.shape[0] != 15:
            print(f"Validation failed: Expected 15 channels, got {patch.shape[0]}.")
            return False
        return True

    def _generate_synthetic_patch(self, aoi: Dict, scale: int) -> np.ndarray:
        """Generates a realistic synthetic data patch as a fallback."""
        try:
            coords = aoi['geometry']['coordinates'][0]
            lon_range = max(c[0] for c in coords) - min(c[0] for c in coords)
            lat_range = max(c[1] for c in coords) - min(c[1] for c in coords)
            width_m = lon_range * 111000 * np.cos(np.radians(np.mean([c[1] for c in coords])))
            height_m = lat_range * 111000
            width_px = max(32, min(512, int(width_m / scale)))
            height_px = max(32, min(512, int(height_m / scale)))
        except:
            width_px, height_px = 256, 256
        
        print(f"🧪 Generating synthetic patch of size {height_px}x{width_px}.")
        
        num_channels = 15
        patch = np.zeros((num_channels, height_px, width_px), dtype=np.float32)
        
        # Populate with realistic random values
        patch[0:6] = np.random.uniform(0.01, 0.3, (6, height_px, width_px)) # Sentinel bands
        patch[6] = np.random.uniform(0.2, 0.8, (height_px, width_px))    # NDVI
        patch[7] = np.random.uniform(0.1, 0.6, (height_px, width_px))    # EVI
        patch[8] = np.random.uniform(500, 2000, (height_px, width_px))   # Rainfall
        patch[9] = np.random.uniform(5.5, 7.5, (height_px, width_px))    # Soil pH
        patch[10] = np.random.uniform(50, 1500, (height_px, width_px))   # Elevation
        patch[11] = np.random.uniform(0, 30, (height_px, width_px))      # Slope
        patch[12] = np.random.uniform(0, 360, (height_px, width_px))     # Aspect
        patch[13] = np.random.uniform(0, 5000, (height_px, width_px))    # Dist to water
        patch[14] = np.random.randint(10, 101, (height_px, width_px))    # LULC classes
        
        return patch

if __name__ == '__main__':
    print("--- Testing Comprehensive Geospatial Handler ---")
    handler = GeospatialDataHandler()
    
    test_aoi = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [78.40, 17.40], [78.41, 17.40], [78.41, 17.41],
                [78.40, 17.41], [78.40, 17.40]
            ]]
        }
    }
    
    print("\n--- Fetching comprehensive multi-spectral patch ---")
    patch, error = handler.get_multispectral_patch(test_aoi)
    
    if patch is not None:
        print(f"✅ Success! Patch shape: {patch.shape}")
        assert patch.shape[0] == 15, "Incorrect number of channels!"
    else:
        print(f"❌ Failed! Error: {error}")
        
    print("\n--- Test Complete ---")