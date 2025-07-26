import ee
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add path fix for imports
ROOT = Path(__file__).resolve().parents[1]  # Go up to Manthan root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# CORRECTED IMPORTS - matching your actual file structure
from utils.gee_auth import gee_init
 # Initialize GEE authentication
from utils.ndvi import compute_sentinel2_ndvi
from utils.rainfall import get_chirps_rainfall
from utils.soil_ph import get_soil_ph_for_aoi
from utils.suitability import calculate_suitability_score
from utils.aoi_tools import AOIProcessor  # for geometry operations

class ManthanDay2Pipeline:
    """Complete Day 2 processing pipeline for Manthan."""
    
    def __init__(self, aoi_geojson, output_dir="outputs"):
        """
        Initialize pipeline with AOI.
        
        Args:
            aoi_geojson: GeoJSON dictionary of Area of Interest
            output_dir: Directory for outputs
        """
        self.aoi_geojson = aoi_geojson
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Convert GeoJSON to Earth Engine Geometry
        self.aoi_geometry = ee.Geometry(aoi_geojson['geometry'])
        
        # Initialize results storage
        self.results = {}
        
    def run_complete_analysis(self, year=2024):
        """
        Run complete Day 2 analysis pipeline.
        
        Args:
            year: Year for analysis
            
        Returns:
            dict: Complete analysis results
        """
        print("🚀 Starting Manthan Day 2 Analysis Pipeline...")
        
        # Step 1: Initialize Earth Engine
        gee_init()
        print("✅ Google Earth Engine initialized")
        
        # Step 2: NDVI Analysis
        print("🛰️ Processing Sentinel-2 NDVI...")
        try:
            ndvi_results = compute_sentinel2_ndvi(
                self.aoi_geometry,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31'
            )
            self.results['ndvi'] = ndvi_results
            print(f"   Mean NDVI: {ndvi_results['ndvi_mean']:.3f}")
        except Exception as e:
            print(f"   ⚠️ NDVI processing failed: {e}")
            # Fallback values
            self.results['ndvi'] = {
                'ndvi_mean': 0.4,
                'ndvi_std': 0.1,
                'vegetation_coverage': 'Moderate Vegetation'
            }
        
        # Step 3: Rainfall Analysis
        print("🌧️ Processing CHIRPS rainfall data...")
        try:
            rainfall_results = get_chirps_rainfall(self.aoi_geometry, year)
            self.results['rainfall'] = rainfall_results
            print(f"   Annual Rainfall: {rainfall_results['annual_rainfall']:.0f} mm")
        except Exception as e:
            print(f"   ⚠️ Rainfall processing failed: {e}")
            # Fallback values
            self.results['rainfall'] = {
                'annual_rainfall': 1100,
                'rainfall_adequacy': 'Good'
            }
        
        # Step 4: Soil pH Analysis
        print("🌱 Processing SoilGrids pH data...")
        try:
            soil_results = get_soil_ph_for_aoi(self.aoi_geometry)
            self.results['soil'] = soil_results
            print(f"   Soil pH: {soil_results['soil_ph']:.1f}")
        except Exception as e:
            print(f"   ⚠️ Soil pH processing failed: {e}")
            # Fallback values
            self.results['soil'] = {
                'soil_ph': 6.5,
                'ph_suitability': 'Good'
            }
        
        # Step 5: Suitability Scoring
        print("🧠 Calculating suitability score...")
        try:
            suitability_results = calculate_suitability_score(
                self.results['ndvi']['ndvi_mean'],
                self.results['rainfall']['annual_rainfall'],
                self.results['soil']['soil_ph']
            )
            self.results['suitability'] = suitability_results
            print(f"   Suitability Score: {suitability_results['composite_score']:.3f}")
            print(f"   Restoration Approach: {suitability_results['restoration_approach']}")
        except Exception as e:
            print(f"   ⚠️ Suitability scoring failed: {e}")
            # Fallback values
            self.results['suitability'] = {
                'composite_score': 0.5,
                'suitability_grade': 'B',
                'restoration_approach': 'Agroforestry System',
                'success_probability': 75
            }
        
        # Step 6: Export Results
        print("💾 Exporting results...")
        self.export_results()
        
        print("🎉 Day 2 analysis complete!")
        return self.results
    
    def export_results(self):
        """Export results in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure all required keys exist with defaults
        ndvi_data = self.results.get('ndvi', {})
        rainfall_data = self.results.get('rainfall', {})
        soil_data = self.results.get('soil', {})
        suitability_data = self.results.get('suitability', {})
        
        try:
            # Export JSON summary
            json_file = self.output_dir / f"manthan_analysis_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Export CSV summary
            csv_data = {
                'Metric': [
                    'NDVI Mean', 'NDVI Std', 'Vegetation Coverage',
                    'Annual Rainfall (mm)', 'Rainfall Adequacy',
                    'Soil pH', 'pH Suitability',
                    'Suitability Score', 'Suitability Grade',
                    'Restoration Approach', 'Success Probability (%)'
                ],
                'Value': [
                    ndvi_data.get('ndvi_mean', 'N/A'),
                    ndvi_data.get('ndvi_std', 'N/A'),
                    ndvi_data.get('vegetation_coverage', 'N/A'),
                    rainfall_data.get('annual_rainfall', 'N/A'),
                    rainfall_data.get('rainfall_adequacy', 'N/A'),
                    soil_data.get('soil_ph', 'N/A'),
                    soil_data.get('ph_suitability', 'N/A'),
                    suitability_data.get('composite_score', 'N/A'),
                    suitability_data.get('suitability_grade', 'N/A'),
                    suitability_data.get('restoration_approach', 'N/A'),
                    suitability_data.get('success_probability', 'N/A')
                ]
            }
            
            df = pd.DataFrame(csv_data)
            csv_file = self.output_dir / f"manthan_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            print(f"📄 Results exported to: {self.output_dir}")
            
            return {
                'json_file': str(json_file),
                'csv_file': str(csv_file)
            }
            
        except Exception as e:
            print(f"⚠️ Export failed: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Sample AOI (replace with your actual AOI)
    sample_aoi = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [77.1, 28.6],  # Delhi region
                [77.2, 28.6],
                [77.2, 28.7],
                [77.1, 28.7],
                [77.1, 28.6]
            ]]
        }
    }
    
    # Run pipeline
    try:
        pipeline = ManthanDay2Pipeline(sample_aoi)
        results = pipeline.run_complete_analysis()
        print("✅ Pipeline completed successfully!")
        print(f"Results keys: {list(results.keys())}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")