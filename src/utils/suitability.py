# src/utils/suitability.py
def calculate_suitability_score(ndvi_mean, rainfall_annual, soil_ph):
    """
    Calculate forest restoration suitability score.
    
    Args:
        ndvi_mean: Mean NDVI value (0-1)
        rainfall_annual: Annual rainfall in mm
        soil_ph: Soil pH value
    
    Returns:
        dict: Suitability analysis results
    """
    # Normalize inputs
    ndvi_norm = max(0, min(1, ndvi_mean))
    ph_norm = soil_ph / 7.0
    rain_norm = rainfall_annual / 1200.0
    
    # Calculate composite score
    score = 0.5 * ndvi_norm + 0.3 * ph_norm + 0.2 * rain_norm
    score = max(0, min(1, score))
    
    # Classify restoration approach
    if score >= 0.65:
        approach = "Miyawaki Dense Forest"
        grade = "A"
        success_prob = 85 + int(score * 15)
    elif score >= 0.40:
        approach = "Agroforestry System"
        grade = "B"
        success_prob = 70 + int(score * 15)
    else:
        approach = "Basic Revegetation"
        grade = "C"
        success_prob = 50 + int(score * 20)
    
    return {
        'composite_score': round(score, 3),
        'suitability_grade': grade,
        'restoration_approach': approach,
        'success_probability': success_prob
    }
