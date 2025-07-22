"""
Forest Restoration Suitability Engine
====================================
AI-powered analysis engine for determining optimal restoration strategies
and native species recommendations based on environmental conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class SuitabilityEngine:
    """Calculates restoration suitability and provides recommendations."""
    
    def __init__(self):
        """Initialize the suitability engine."""
        self.load_species_database()
        self.scoring_weights = {
            'ndvi': 0.40,
            'rainfall': 0.30,
            'soil_ph': 0.20,
            'slope': 0.10
        }
        
    def calculate_composite_score(self,
                                ndvi: float,
                                annual_rainfall: float,
                                soil_ph: float,
                                slope_factor: float = 0.8) -> float:
        """
        Calculate composite suitability score.
        
        Formula: 0.4×NDVI + 0.3×Rainfall + 0.2×pH + 0.1×Slope
        
        Args:
            ndvi: NDVI value (0-1)
            annual_rainfall: Rainfall in mm
            soil_ph: Soil pH value
            slope_factor: Slope suitability (0-1)
            
        Returns:
            Composite score (0-1)
        """
        # Normalize inputs
        ndvi_norm = max(0, min(1, ndvi))
        rainfall_norm = max(0, min(1, annual_rainfall / 1200))  # Normalize to 1200mm
        ph_norm = max(0, min(1, self._normalize_ph(soil_ph)))
        slope_norm = max(0, min(1, slope_factor))
        
        # Calculate weighted score
        score = (
            self.scoring_weights['ndvi'] * ndvi_norm +
            self.scoring_weights['rainfall'] * rainfall_norm +
            self.scoring_weights['soil_ph'] * ph_norm +
            self.scoring_weights['slope'] * slope_norm
        )
        
        return round(score, 4)
    
    def classify_restoration_approach(self, score: float) -> str:
        """
        Classify optimal restoration approach based on suitability score.
        
        Args:
            score: Composite suitability score
            
        Returns:
            Restoration approach classification
        """
        if score >= 0.75:
            return "Miyawaki Dense Forest"
        elif score >= 0.60:
            return "Agroforestry System"
        elif score >= 0.45:
            return "Eco-Tourism Forest"
        elif score >= 0.30:
            return "Basic Revegetation"
        else:
            return "Soil Improvement Required"
    
    def get_detailed_assessment(self, 
                              ndvi: float,
                              rainfall: float,
                              soil_ph: float,
                              area_ha: float) -> Dict:
        """
        Get comprehensive suitability assessment.
        
        Args:
            ndvi: NDVI value
            rainfall: Annual rainfall (mm)
            soil_ph: Soil pH
            area_ha: Area in hectares
            
        Returns:
            Detailed assessment dictionary
        """
        # Calculate main score
        score = self.calculate_composite_score(ndvi, rainfall, soil_ph)
        approach = self.classify_restoration_approach(score)
        
        # Analyze individual factors
        factor_analysis = self._analyze_individual_factors(ndvi, rainfall, soil_ph)
        
        # Calculate implementation metrics
        implementation = self._calculate_implementation_metrics(approach, area_ha)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(approach, factor_analysis)
        
        return {
            'composite_score': score,
            'restoration_approach': approach,
            'suitability_grade': self._get_suitability_grade(score),
            'factor_analysis': factor_analysis,
            'implementation_metrics': implementation,
            'recommendations': recommendations,
            'success_probability': self._estimate_success_probability(score, factor_analysis)
        }
    
    def recommend_species(self,
                         approach: str,
                         soil_ph: float,
                         rainfall: float,
                         region: str = "Western_Ghats") -> Dict[str, List[Dict]]:
        """
        Recommend native species based on restoration approach and conditions.
        
        Args:
            approach: Restoration approach
            soil_ph: Soil pH value
            rainfall: Annual rainfall
            region: Geographic region
            
        Returns:
            Species recommendations organized by forest layers
        """
        # Filter species by environmental compatibility
        compatible_species = self._filter_species_by_environment(soil_ph, rainfall, region)
        
        # Organize by restoration approach
        if approach == "Miyawaki Dense Forest":
            return self._get_miyawaki_species_plan(compatible_species)
        elif approach == "Agroforestry System":
            return self._get_agroforestry_species_plan(compatible_species)
        elif approach == "Eco-Tourism Forest":
            return self._get_ecotourism_species_plan(compatible_species)
        else:
            return self._get_basic_restoration_species_plan(compatible_species)
    
    def load_species_database(self):
        """Load native species database."""
        # Comprehensive species database for Indian forest restoration
        self.species_database = {
            "canopy_trees": [
                {
                    "scientific_name": "Terminalia bellirica",
                    "common_name": "Bahera",
                    "family": "Combretaceae",
                    "native_regions": ["Western_Ghats", "Eastern_Ghats", "Central_India"],
                    "growth_rate": "Fast",
                    "max_height": 30,
                    "soil_ph_min": 6.0,
                    "soil_ph_max": 7.5,
                    "rainfall_min": 800,
                    "rainfall_max": 2000,
                    "ecological_functions": ["Carbon sequestration", "Soil conservation", "Wildlife habitat"],
                    "economic_value": "High",
                    "conservation_status": "Least Concern"
                },
                {
                    "scientific_name": "Tectona grandis",
                    "common_name": "Teak",
                    "family": "Lamiaceae",
                    "native_regions": ["Western_Ghats", "Central_India"],
                    "growth_rate": "Medium",
                    "max_height": 40,
                    "soil_ph_min": 6.5,
                    "soil_ph_max": 7.5,
                    "rainfall_min": 1200,
                    "rainfall_max": 2500,
                    "ecological_functions": ["Timber production", "Carbon storage"],
                    "economic_value": "Very High",
                    "conservation_status": "Near Threatened"
                },
                {
                    "scientific_name": "Shorea robusta",
                    "common_name": "Sal",
                    "family": "Dipterocarpaceae",
                    "native_regions": ["Eastern_India", "Central_India"],
                    "growth_rate": "Medium",
                    "max_height": 35,
                    "soil_ph_min": 5.5,
                    "soil_ph_max": 7.0,
                    "rainfall_min": 1000,
                    "rainfall_max": 1800,
                    "ecological_functions": ["Forest ecosystem keystone", "Wildlife habitat"],
                    "economic_value": "High",
                    "conservation_status": "Least Concern"
                }
            ],
            "sub_canopy_trees": [
                {
                    "scientific_name": "Lagerstroemia speciosa",
                    "common_name": "Pride of India",
                    "family": "Lythraceae",
                    "native_regions": ["Western_Ghats", "Eastern_Ghats"],
                    "growth_rate": "Medium",
                    "max_height": 20,
                    "soil_ph_min": 6.0,
                    "soil_ph_max": 7.0,
                    "rainfall_min": 600,
                    "rainfall_max": 1500,
                    "ecological_functions": ["Pollinator support", "Ornamental value"],
                    "economic_value": "Medium",
                    "conservation_status": "Least Concern"
                },
                {
                    "scientific_name": "Cassia fistula",
                    "common_name": "Golden Shower",
                    "family": "Fabaceae",
                    "native_regions": ["Western_Ghats", "Central_India", "Eastern_India"],
                    "growth_rate": "Fast",
                    "max_height": 15,
                    "soil_ph_min": 6.0,
                    "soil_ph_max": 8.0,
                    "rainfall_min": 500,
                    "rainfall_max": 1200,
                    "ecological_functions": ["Nitrogen fixation", "Ornamental", "Medicinal"],
                    "economic_value": "Medium",
                    "conservation_status": "Least Concern"
                }
            ],
            "understory": [
                {
                    "scientific_name": "Bambusa bambos",
                    "common_name": "Indian Thorny Bamboo",
                    "family": "Poaceae",
                    "native_regions": ["Western_Ghats", "Eastern_Ghats", "Central_India"],
                    "growth_rate": "Very Fast",
                    "max_height": 15,
                    "soil_ph_min": 5.5,
                    "soil_ph_max": 7.0,
                    "rainfall_min": 800,
                    "rainfall_max": 2500,
                    "ecological_functions": ["Rapid carbon sequestration", "Erosion control"],
                    "economic_value": "High",
                    "conservation_status": "Least Concern"
                },
                {
                    "scientific_name": "Calamus rotang",
                    "common_name": "Rattan Palm",
                    "family": "Arecaceae",
                    "native_regions": ["Western_Ghats", "Eastern_Ghats"],
                    "growth_rate": "Medium",
                    "max_height": 10,
                    "soil_ph_min": 5.0,
                    "soil_ph_max": 6.5,
                    "rainfall_min": 1200,
                    "rainfall_max": 3000,
                    "ecological_functions": ["Understory habitat", "Climbing support"],
                    "economic_value": "Medium",
                    "conservation_status": "Vulnerable"
                }
            ],
            "ground_cover": [
                {
                    "scientific_name": "Ixora coccinea",
                    "common_name": "Flame of the Woods",
                    "family": "Rubiaceae",
                    "native_regions": ["Western_Ghats", "Southern_India"],
                    "growth_rate": "Medium",
                    "max_height": 3,
                    "soil_ph_min": 5.5,
                    "soil_ph_max": 6.5,
                    "rainfall_min": 800,
                    "rainfall_max": 2000,
                    "ecological_functions": ["Pollinator support", "Ground cover"],
                    "economic_value": "Low",
                    "conservation_status": "Least Concern"
                }
            ]
        }
    
    def _normalize_ph(self, ph: float) -> float:
        """Normalize pH to 0-1 scale with optimal range 6.5-7.0."""
        if 6.5 <= ph <= 7.0:
            return 1.0
        elif 6.0 <= ph < 6.5 or 7.0 < ph <= 7.5:
            return 0.8
        elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
            return 0.6
        elif 5.0 <= ph < 5.5 or 8.0 < ph <= 8.5:
            return 0.4
        else:
            return 0.2
    
    def _analyze_individual_factors(self, ndvi: float, rainfall: float, soil_ph: float) -> Dict:
        """Analyze individual environmental factors."""
        return {
            'vegetation_status': {
                'value': ndvi,
                'rating': self._rate_ndvi(ndvi),
                'description': self._describe_ndvi(ndvi)
            },
            'water_availability': {
                'value': rainfall,
                'rating': self._rate_rainfall(rainfall),
                'description': self._describe_rainfall(rainfall)
            },
            'soil_condition': {
                'value': soil_ph,
                'rating': self._rate_ph(soil_ph),
                'description': self._describe_ph(soil_ph)
            }
        }
    
    def _rate_ndvi(self, ndvi: float) -> str:
        """Rate NDVI value."""
        if ndvi >= 0.7: return "Excellent"
        elif ndvi >= 0.5: return "Good"
        elif ndvi >= 0.3: return "Fair"
        else: return "Poor"
    
    def _describe_ndvi(self, ndvi: float) -> str:
        """Describe NDVI condition."""
        if ndvi >= 0.7: return "Dense, healthy vegetation present"
        elif ndvi >= 0.5: return "Moderate vegetation cover"
        elif ndvi >= 0.3: return "Sparse vegetation, some regeneration needed"
        else: return "Minimal vegetation, significant restoration required"
    
    def _rate_rainfall(self, rainfall: float) -> str:
        """Rate rainfall adequacy."""
        if rainfall >= 1200: return "Excellent"
        elif rainfall >= 800: return "Good"
        elif rainfall >= 600: return "Fair"
        else: return "Poor"
    
    def _describe_rainfall(self, rainfall: float) -> str:
        """Describe rainfall condition."""
        if rainfall >= 1500: return "High rainfall, excellent for forest growth"
        elif rainfall >= 1200: return "Adequate rainfall for most forest species"
        elif rainfall >= 800: return "Moderate rainfall, suitable for hardy species"
        else: return "Low rainfall, drought-tolerant species recommended"
    
    def _rate_ph(self, ph: float) -> str:
        """Rate soil pH."""
        if 6.5 <= ph <= 7.0: return "Excellent"
        elif 6.0 <= ph <= 7.5: return "Good"
        elif 5.5 <= ph <= 8.0: return "Fair"
        else: return "Poor"
    
    def _describe_ph(self, ph: float) -> str:
        """Describe pH condition."""
        if ph < 5.5: return "Acidic soil, lime amendment may be needed"
        elif 5.5 <= ph < 6.5: return "Slightly acidic, good for acid-loving plants"
        elif 6.5 <= ph <= 7.0: return "Optimal pH for most forest species"
        elif 7.0 < ph <= 7.5: return "Slightly alkaline, suitable for most plants"
        else: return "Alkaline soil, may need acidification"
    
    def _get_suitability_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.90: return "A+"
        elif score >= 0.80: return "A"
        elif score >= 0.70: return "B+"
        elif score >= 0.60: return "B"
        elif score >= 0.50: return "C+"
        elif score >= 0.40: return "C"
        else: return "D"
    
    def _calculate_implementation_metrics(self, approach: str, area_ha: float) -> Dict:
        """Calculate implementation metrics."""
        metrics_data = {
            "Miyawaki Dense Forest": {
                "tree_density_per_ha": 2500,
                "species_diversity": 15,
                "establishment_years": 3,
                "cost_per_ha": 150000
            },
            "Agroforestry System": {
                "tree_density_per_ha": 400,
                "species_diversity": 8,
                "establishment_years": 5,
                "cost_per_ha": 80000
            },
            "Eco-Tourism Forest": {
                "tree_density_per_ha": 800,
                "species_diversity": 12,
                "establishment_years": 4,
                "cost_per_ha": 120000
            },
            "Basic Revegetation": {
                "tree_density_per_ha": 600,
                "species_diversity": 5,
                "establishment_years": 6,
                "cost_per_ha": 50000
            }
        }
        
        base_metrics = metrics_data.get(approach, metrics_data["Basic Revegetation"])
        
        return {
            'total_trees_needed': int(base_metrics["tree_density_per_ha"] * area_ha),
            'recommended_species_count': base_metrics["species_diversity"],
            'establishment_timeline_years': base_metrics["establishment_years"],
            'estimated_cost_inr': int(base_metrics["cost_per_ha"] * area_ha),
            'maintenance_period_years': 3
        }
    
    def _generate_recommendations(self, approach: str, factor_analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Approach-specific recommendations
        if approach == "Miyawaki Dense Forest":
            recommendations.extend([
                "Plant native species in high density (2500 trees/hectare)",
                "Use multi-layer planting: canopy, sub-canopy, understory, and ground cover",
                "Apply thick mulch layer for moisture retention and weed suppression",
                "Monitor intensively for first 3 years until forest becomes self-sustaining"
            ])
        elif approach == "Agroforestry System":
            recommendations.extend([
                "Integrate trees with agricultural crops for economic benefits",
                "Plant boundary trees first, then gradually introduce inter-cropping",
                "Select fast-growing timber and fruit species for income generation",
                "Plan for seasonal crop rotations between tree rows"
            ])
        
        # Factor-specific recommendations
        vegetation_rating = factor_analysis['vegetation_status']['rating']
        if vegetation_rating in ['Poor', 'Fair']:
            recommendations.append("Consider soil preparation and organic matter addition before planting")
        
        rainfall_rating = factor_analysis['water_availability']['rating']
        if rainfall_rating in ['Poor', 'Fair']:
            recommendations.extend([
                "Install rainwater harvesting systems",
                "Select drought-tolerant native species",
                "Plan planting to coincide with monsoon season"
            ])
        
        soil_rating = factor_analysis['soil_condition']['rating']
        if soil_rating in ['Poor', 'Fair']:
            recommendations.append("Conduct detailed soil testing and apply appropriate amendments")
        
        return recommendations
    
    def _estimate_success_probability(self, score: float, factor_analysis: Dict) -> int:
        """Estimate restoration success probability as percentage."""
        base_probability = int(score * 85)  # Base on composite score
        
        # Adjust based on critical factors
        rainfall_rating = factor_analysis['water_availability']['rating']
        if rainfall_rating == 'Poor':
            base_probability -= 15
        elif rainfall_rating == 'Fair':
            base_probability -= 5
        
        return max(20, min(95, base_probability))  # Clamp between 20-95%
    
    def _filter_species_by_environment(self, ph: float, rainfall: float, region: str) -> Dict:
        """Filter species database by environmental conditions."""
        filtered_species = {}
        
        for category, species_list in self.species_database.items():
            suitable_species = []
            
            for species in species_list:
                # Check pH compatibility
                if species['soil_ph_min'] <= ph <= species['soil_ph_max']:
                    # Check rainfall compatibility
                    if species['rainfall_min'] <= rainfall <= species['rainfall_max']:
                        # Check regional compatibility
                        if region in species['native_regions'] or 'All_India' in species.get('native_regions', []):
                            suitable_species.append(species)
            
            filtered_species[category] = suitable_species
        
        return filtered_species
    
    def _get_miyawaki_species_plan(self, compatible_species: Dict) -> Dict[str, List[Dict]]:
        """Get Miyawaki forest species plan."""
        return {
            'canopy_layer': compatible_species.get('canopy_trees', [])[:3],
            'sub_canopy_layer': compatible_species.get('sub_canopy_trees', [])[:4],
            'understory_layer': compatible_species.get('understory', [])[:5],
            'ground_cover_layer': compatible_species.get('ground_cover', [])[:6]
        }
    
    def _get_agroforestry_species_plan(self, compatible_species: Dict) -> Dict[str, List[Dict]]:
        """Get agroforestry species plan."""
        return {
            'timber_trees': [s for s in compatible_species.get('canopy_trees', []) if s['economic_value'] in ['High', 'Very High']][:2],
            'fruit_trees': compatible_species.get('sub_canopy_trees', [])[:3],
            'boundary_trees': compatible_species.get('understory', [])[:3],
            'nitrogen_fixers': [s for s in compatible_species.get('sub_canopy_trees', []) if 'Nitrogen fixation' in s.get('ecological_functions', [])][:2]
        }
    
    def _get_ecotourism_species_plan(self, compatible_species: Dict) -> Dict[str, List[Dict]]:
        """Get eco-tourism forest species plan."""
        return {
            'shade_trees': compatible_species.get('canopy_trees', [])[:2],
            'flowering_trees': compatible_species.get('sub_canopy_trees', [])[:3],
            'ornamental_plants': compatible_species.get('ground_cover', [])[:4],
            'wildlife_attractors': compatible_species.get('understory', [])[:3]
        }
    
    def _get_basic_restoration_species_plan(self, compatible_species: Dict) -> Dict[str, List[Dict]]:
        """Get basic restoration species plan."""
        return {
            'pioneer_species': [s for s in compatible_species.get('sub_canopy_trees', []) if s['growth_rate'] == 'Fast'][:3],
            'soil_stabilizers': compatible_species.get('understory', [])[:3],
            'hardy_ground_cover': compatible_species.get('ground_cover', [])[:3]
        }

# Test function
def main():
    """Test the suitability engine."""
    engine = SuitabilityEngine()
    
    # Test with sample data
    test_ndvi = 0.65
    test_rainfall = 1100
    test_ph = 6.8
    test_area = 25.5
    
    # Get detailed assessment
    assessment = engine.get_detailed_assessment(test_ndvi, test_rainfall, test_ph, test_area)
    
    print("🌱 Suitability Assessment Results:")
    print(f"Composite Score: {assessment['composite_score']}")
    print(f"Grade: {assessment['suitability_grade']}")
    print(f"Restoration Approach: {assessment['restoration_approach']}")
    print(f"Success Probability: {assessment['success_probability']}%")
    
    print(f"\n📊 Factor Analysis:")
    for factor, analysis in assessment['factor_analysis'].items():
        print(f"{factor.replace('_', ' ').title()}: {analysis['rating']} - {analysis['description']}")
    
    print(f"\n🏗️ Implementation Metrics:")
    metrics = assessment['implementation_metrics']
    print(f"Total trees needed: {metrics['total_trees_needed']:,}")
    print(f"Estimated cost: ₹{metrics['estimated_cost_inr']:,}")
    print(f"Timeline: {metrics['establishment_timeline_years']} years")
    
    print(f"\n💡 Key Recommendations:")
    for i, rec in enumerate(assessment['recommendations'][:3], 1):
        print(f"{i}. {rec}")
    
    # Test species recommendations
    species_plan = engine.recommend_species(
        assessment['restoration_approach'], 
        test_ph, 
        test_rainfall
    )
    
    print(f"\n🌳 Recommended Species:")
    for layer, species_list in species_plan.items():
        if species_list:
            print(f"\n{layer.replace('_', ' ').title()}:")
            for species in species_list[:2]:  # Show first 2
                print(f"  - {species['common_name']} ({species['scientific_name']})")

if __name__ == "__main__":
    main()
