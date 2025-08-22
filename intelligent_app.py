# intelligent_app.py

import json
import argparse
from manthan_core.schemas.aoi import AOI
from manthan_core.site_assessment.gee_pipeline import build_site_fingerprint
from manthan_core.recommender.rules_engine import recommend_species
from manthan_core.forecasting.lgbm_model import forecast_outcomes

def run_manthan_pipeline(aoi_path: str):
    """Orchestrates the three phases of the Manthan pipeline."""
    # Load AOI from file
    with open(aoi_path, 'r') as f:
        aoi_data = json.load(f)
    aoi = AOI(**aoi_data)

    # Phase 1: Site Assessment
    fingerprint = build_site_fingerprint(aoi)
    
    # Phase 2: Species Recommendation
    planting_plan = recommend_species(fingerprint, aoi)
    # Phase 3: Forecasting
    forecast = forecast_outcomes(planting_plan)

    # --- Print Demo Report ---
    print("\n" + "="*50)
    print("      MANTHAN DEMO REPORT")
    print("="*50)
    print(f"\n--- 1. Site Fingerprint for: {fingerprint.aoi_name} ---")
    # --- THIS IS THE CORRECTED LINE ---
    print(fingerprint.model_dump_json(indent=2))
    
    print(f"\n--- 2. Top Species Recommendations ---")
    for rec in planting_plan.recommendations:
        print(f"- {rec.scientific_name} ({rec.common_name}): Score={rec.score:.2f}, Why='{rec.why}'")
        
    print(f"\n--- 3. 3-Year Forecast ---")
    print(f"  - Predicted Survival Rate: {forecast.survival_pct_mean:.1f}%")
    print(f"  - 80% Prediction Interval: {forecast.survival_pct_pi[0]:.1f}% - {forecast.survival_pct_pi[1]:.1f}%")
    print(f"  - Notes: {forecast.notes}")
    print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Manthan Demo Pipeline.")
    parser.add_argument("--aoi", type=str, required=True, help="Path to the AOI GeoJSON file.")
    args = parser.parse_args()
    run_manthan_pipeline(args.aoi)