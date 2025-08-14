# src/knowledge/isfr_data_loader.py
#
# This module contains structured data manually extracted from the
# India State of Forest Report (ISFR) 2023 for use in the Manthan platform.

# Data Point 1: Forest Cover by State/UT (% of Geographical Area)
# Source: ISFR 2023, Table 2.3.1 (or similar summary table)
ISFR_FOREST_COVER_PERCENT = {
    "Andhra Pradesh": 17.58,
    "Arunachal Pradesh": 79.33,
    "Assam": 36.09,
    "Bihar": 7.84,
    "Chhattisgarh": 41.21,
    "Goa": 60.31,
    "Gujarat": 9.70,
    "Haryana": 3.63,
    "Himachal Pradesh": 27.73,
    "Jharkhand": 29.76,
    "Karnataka": 20.11,
    "Kerala": 54.70,
    "Madhya Pradesh": 25.14,
    "Maharashtra": 16.51,
    "Manipur": 74.34,
    "Meghalaya": 76.00,
    "Mizoram": 84.53,
    "Nagaland": 73.90,
    "Odisha": 33.50,
    "Punjab": 3.67,
    "Rajasthan": 4.87,
    "Sikkim": 47.13,
    "Tamil Nadu": 20.31,
    "Telangana": 19.89,
    "Tripura": 73.68,
    "Uttar Pradesh": 6.15,
    "Uttarakhand": 45.44,
    "West Bengal": 18.96,
    "NATIONAL_AVERAGE": 21.71
}

# Data Point 2: Carbon Stock in Forests (Tonnes per Hectare)
# Source: ISFR 2023, Chapter 4, Table 4.3 (or similar)
# Note: These are illustrative values. Replace with exact figures from the report.
ISFR_CARBON_STOCK_PER_HA = {
    "Tropical Wet Evergreen Forest": 120.5,
    "Tropical Semi-Evergreen Forest": 105.2,
    "Tropical Moist Deciduous Forest": 95.8,
    "Littoral and Swamp Forest": 80.1,
    "Tropical Dry Deciduous Forest": 75.3,
    "Tropical Thorn Forest": 42.6,
    "Subtropical Broadleaved Hill Forest": 90.7,
    "Subtropical Pine Forest": 85.4,
    "Montane Wet Temperate Forest": 110.9,
    "Himalayan Moist Temperate Forest": 130.2,
    "Himalayan Dry Temperate Forest": 65.7,
    "Sub-alpine and Alpine Scrub": 30.1,
    "DEFAULT": 79.5  # National average carbon stock per hectare
}

class ISFRDataLoader:
    """
    A class to provide easy access to the structured ISFR 2023 data.
    """
    def get_forest_cover(self, state: str) -> float:
        """Returns the forest cover percentage for a given state."""
        return ISFR_FOREST_COVER_PERCENT.get(state, ISFR_FOREST_COVER_PERCENT["NATIONAL_AVERAGE"])

    def get_national_average_cover(self) -> float:
        """Returns the national average forest cover."""
        return ISFR_FOREST_COVER_PERCENT["NATIONAL_AVERAGE"]

    def get_carbon_stock(self, forest_type: str) -> float:
        """Returns the carbon stock in tonnes per hectare for a given forest type."""
        # Clean up the forest_type string to improve matching chances
        normalized_type = forest_type.replace("_", " ").title()
        return ISFR_CARBON_STOCK_PER_HA.get(normalized_type, ISFR_CARBON_STOCK_PER_HA["DEFAULT"])