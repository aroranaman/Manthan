# FILE: src/core/economic_viability_model.py
import numpy as np
from typing import Dict

class EconomicViabilityModel:
    """
    A model to project the economic viability of an agroforestry or
    restoration project based on the AI framework's parameters.
    """
    def __init__(self):
        """
        Initializes the model with baseline financial assumptions.
        These values can be tuned based on regional data.
        """
        # Costs are based on your framework's data (e.g., ANR, Agroforestry costs)
        self.ESTABLISHMENT_COST_PER_HA = 55000  # Avg. cost for Agroforestry/Silvopasture
        self.ANNUAL_MAINTENANCE_COST_PER_HA = 8000
        
        # Revenue projections based on your framework (₹25,000-75,000/ha/year)
        self.MIN_ANNUAL_RETURN_PER_HA = 25000
        self.MAX_ANNUAL_RETURN_PER_HA = 75000
        
        # Other financial assumptions
        self.YEARS_TO_MATURITY = 3  # Years until significant revenue begins
        self.DISCOUNT_RATE = 0.08   # For future value calculations

    def get_projections(self, area_ha: float, suitability_score: float = 0.8) -> Dict:
        """
        Generates financial projections for a given land parcel.

        Args:
            area_ha (float): The total area of the land parcel in hectares.
            suitability_score (float, optional): A score from 0 to 1 from the AI model,
                                                 used to adjust revenue potential. Defaults to 0.8.

        Returns:
            Dict: A dictionary containing key financial metrics.
        """
        # 1. Calculate Annual Return
        # AI-optimized return based on land suitability
        projected_annual_return = self.MIN_ANNUAL_RETURN_PER_HA + \
            (self.MAX_ANNUAL_RETURN_PER_HA - self.MIN_ANNUAL_RETURN_PER_HA) * suitability_score
        
        total_annual_revenue = projected_annual_return * area_ha
        
        # 2. Calculate Total Costs
        total_establishment_cost = self.ESTABLISHMENT_COST_PER_HA * area_ha
        total_annual_maintenance = self.ANNUAL_MAINTENANCE_COST_PER_HA * area_ha
        
        # 3. Calculate Payback Period
        # Time to recover the initial establishment cost
        net_annual_income_after_maturity = total_annual_revenue - total_annual_maintenance
        
        if net_annual_income_after_maturity <= 0:
            payback_period_years = float('inf') # Project is not profitable
        else:
            payback_period_years = self.YEARS_TO_MATURITY + \
                (total_establishment_cost / net_annual_income_after_maturity)

        # 4. Calculate 10-Year Net Profit
        total_revenue_10_years = 0
        total_maintenance_10_years = 0
        
        for year in range(1, 11):
            total_maintenance_10_years += total_annual_maintenance
            if year > self.YEARS_TO_MATURITY:
                total_revenue_10_years += total_annual_revenue
                
        net_profit_10_years = total_revenue_10_years - (total_establishment_cost + total_maintenance_10_years)

        return {
            "annual_return_per_ha": round(projected_annual_return, 0),
            "payback_period_years": round(payback_period_years, 1) if payback_period_years != float('inf') else "N/A",
            "net_profit_10_years": round(net_profit_10_years, 0),
            "total_establishment_cost": round(total_establishment_cost, 0)
        }

if __name__ == '__main__':
    # --- Smoke Test for the Economic Viability Model ---
    print("--- Initializing Economic Viability Model ---")
    economic_model = EconomicViabilityModel()
    print("✅ Model initialized successfully.")
    
    # Test case: 1.5 hectares of land with high suitability
    test_area_ha = 1.5
    test_suitability = 0.9 # High suitability
    
    print(f"\n--- Generating projections for {test_area_ha} ha with {test_suitability*100}% suitability ---")
    projections = economic_model.get_projections(test_area_ha, test_suitability)
    
    print("\n--- Financial Projections ---")
    print(f"   - Projected Annual Return per Hectare: ₹{projections['annual_return_per_ha']:,.0f}")
    print(f"   - Total Establishment Cost: ₹{projections['total_establishment_cost']:,.0f}")
    print(f"   - Estimated Payback Period: {projections['payback_period_years']} years")
    print(f"   - Projected 10-Year Net Profit: ₹{projections['net_profit_10_years']:,.0f}")
    
    assert projections['annual_return_per_ha'] > 0
    assert projections['payback_period_years'] > 0
    print("\n✅ Smoke test passed successfully.")