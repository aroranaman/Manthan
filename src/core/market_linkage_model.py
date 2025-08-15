# FILE: src/core/market_linkage_model.py
import pandas as pd
import numpy as np
from typing import Dict, List

class MarketLinkageModel:
    """
    An enhanced simulation of the AI-powered market linkage component.
    This model now uses environmental and socioeconomic data inputs to generate
    more realistic predictions for market prices and farmer-buyer matching,
    laying the groundwork for a future ML-driven system.
    """
    def __init__(self):
        """
        Initializes the model with base data. In a real system, this would
        load trained models or connect to a market database.
        """
        self.products = ['Timber (Teak)', 'Medicinal Herbs', 'Fruits (Amla)', 'Bamboo']
        self.buyers = {
            'Pulp & Paper Co.': {'location': 'Nagpur', 'demands': ['Timber (Teak)', 'Bamboo']},
            'Herbal Pharma Inc.': {'location': 'Hyderabad', 'demands': ['Medicinal Herbs', 'Fruits (Amla)']},
            'Organic Foods Ltd.': {'location': 'Pune', 'demands': ['Fruits (Amla)']},
            'Sustainable Crafts LLC': {'location': 'Jaipur', 'demands': ['Bamboo']}
        }
        self.base_prices = {
            'Timber (Teak)': 8000,
            'Medicinal Herbs': 500,
            'Fruits (Amla)': 60,
            'Bamboo': 2500
        }

    def get_price_prediction(self, product: str, climate_data: Dict, market_proximity: float) -> Dict:
        """
        Simulates an AI-driven price prediction using environmental and socioeconomic inputs.

        Args:
            product (str): The agroforestry product to be priced.
            climate_data (Dict): A dictionary containing climate info like 'mean_temperature'
                                 and 'annual_precipitation'.
            market_proximity (float): A score from 0 (far) to 1 (close) representing
                                      proximity to major markets.

        Returns:
            Dict: A dictionary with the predicted price and market trend.
        """
        base_price = self.base_prices.get(product, 0)
        
        # 1. Climate Quality Multiplier (simulated)
        # Ideal conditions lead to higher quality products and better prices.
        # This simulates the model learning optimal growing conditions.
        climate_multiplier = 1.0
        if 'Teak' in product and climate_data.get('annual_precipitation', 1000) > 1200:
            climate_multiplier = 1.15 # Teak thrives in moist conditions
        elif 'Amla' in product and climate_data.get('mean_temperature', 25) > 28:
            climate_multiplier = 1.1 # Amla prefers warmer climates
        
        # 2. Market Proximity Multiplier
        # Closer proximity reduces logistics costs, increasing the farm-gate price.
        proximity_multiplier = 1.0 + (0.1 * market_proximity) # Up to 10% price increase for proximity
        
        # Final Price Calculation
        predicted_price = base_price * climate_multiplier * proximity_multiplier * (1 + np.random.uniform(-0.05, 0.05))
        
        trend = np.random.choice(['Stable', 'Rising', 'Falling'])
        
        return {
            "product": product,
            "predicted_price_per_unit": round(predicted_price, 2),
            "market_trend": trend
        }

    def match_farmer_to_buyers(self, products_grown: List[str], land_use_data: Dict) -> pd.DataFrame:
        """
        Simulates an AI matching algorithm using land use and location data.

        Args:
            products_grown (List[str]): A list of products the farmer is growing.
            land_use_data (Dict): A dictionary with info like 'ownership_type' and 'market_proximity'.

        Returns:
            pd.DataFrame: A DataFrame of matched buyers and their requirements.
        """
        matches = []
        for product in products_grown:
            potential_buyers = [b for b, d in self.buyers.items() if product in d['demands']]
            
            if not potential_buyers:
                continue

            # Simple logic: assume the closest buyer is the best match for now.
            # A real model would use a more complex matching score.
            best_buyer = np.random.choice(potential_buyers)
            
            # Larger land holdings can fulfill higher demand
            demand_level = 'High' if land_use_data.get('area_ha', 1) > 10 else np.random.choice(['Medium', 'Low'])
            
            matches.append({
                "Product": product,
                "Matched Buyer": best_buyer,
                "Demand Level": demand_level,
                "Est. Offtake (tons/year)": np.random.randint(5, 50)
            })
            
        return pd.DataFrame(matches)

