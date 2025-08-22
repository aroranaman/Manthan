from manthan_core.schemas.species_rec import PlantingPlan
from manthan_core.schemas.forecast import ForecastBundle

def forecast_outcomes(plan: PlantingPlan) -> ForecastBundle:
    """Forecasts long-term outcomes using a baseline LightGBM model."""
    print("INFO: Forecasting outcomes...")
    # TODO: Load pre-trained LightGBM quantile models and predict
    
    # Placeholder forecast for the demo
    forecast = ForecastBundle(
        survival_pct_mean=75.0,
        survival_pct_pi=(65.0, 85.0),
        notes="Forecast based on baseline model. High uncertainty."
    )
    print("INFO: Successfully forecasted outcomes.")
    return forecast
