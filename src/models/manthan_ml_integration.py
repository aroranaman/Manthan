"""
Manthan ML Integration Layer
Location: src/models/manthan_ml_integration.py
Integrates ML models with existing Streamlit app - replaces mock data with real AI
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Add paths for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from src.models.ncf_species import get_ai_species_recommendations
    from knowledge.ecological_graph import get_ecological_context, quick_zone_lookup
    ML_MODELS_LOADED = True
    print("✅ Successfully imported ML models")
except ImportError as e:
    print(f"⚠️  ML models not loaded: {e}")
    ML_MODELS_LOADED = False

class ManthanAIIntegration:
    """Main integration class for Manthan AI features"""
    
    def __init__(self):
        self.models_available = ML_MODELS_LOADED
        print(f"🧠 Manthan AI Integration initialized. Models available: {self.models_available}")
        
    def enhance_analysis_results(self, analysis_results: Dict, 
                                coordinates: Optional[Tuple[float, float]] = None,
                                aoi_data: Optional[Dict] = None) -> Dict:
        """
        Enhance existing analysis results with AI recommendations
        
        This function integrates with your existing day2_pipeline results
        and replaces mock data in Streamlit with real AI recommendations
        """
        
        print(f"🔄 Enhancing analysis results. Coordinates: {coordinates}")
        
        if not self.models_available:
            print("⚠️  Using fallback recommendations")
            return self._fallback_recommendations(analysis_results, coordinates)
        
        try:
            # Extract environmental conditions from existing analysis
            # Handle both new and old result formats (ndvi_data vs ndvi, etc.)
            ndvi_data = analysis_results.get('ndvi_data', analysis_results.get('ndvi', {}))
            rain_data = analysis_results.get('environmental_data', analysis_results.get('rainfall', {}))
            soil_data = analysis_results.get('soil_data', analysis_results.get('soil', {}))
            
            environmental_conditions = {
                'ndvi': ndvi_data.get('ndvi_mean', 0.5),
                'rainfall': rain_data.get('annual_rainfall', 1000),
                'soil_ph': soil_data.get('soil_ph', 6.5),
                'temperature': 27.0,  # Default - enhance with real data later
                'elevation': 300,     # Default - enhance with real data later
            }
            
            print(f"📊 Environmental conditions: {environmental_conditions}")
            
            # Get biogeographic zone if coordinates available
            biogeographic_zone = 'Unknown'
            if coordinates:
                lat, lon = coordinates
                biogeographic_zone = quick_zone_lookup(lat, lon)
                environmental_conditions['biogeographic_zone'] = biogeographic_zone
                print(f"🌍 Mapped to biogeographic zone: {biogeographic_zone}")
            
            # Get AI species recommendations
            species_data = get_ai_species_recommendations(analysis_results, coordinates)
            print(f"🌿 Generated {len(species_data['species_recommendations'])} species recommendations")
            
            # Get ecological context if coordinates available
            ecological_context = {}
            if coordinates:
                lat, lon = coordinates
                try:
                    ecological_context = get_ecological_context(lat, lon, environmental_conditions)
                    print(f"🌳 Ecological context: {ecological_context.get('biogeographic_zone', 'Unknown')}")
                except Exception as e:
                    print(f"⚠️  Ecological context failed: {e}")
                    ecological_context = {'biogeographic_zone': biogeographic_zone}
            
            # Calculate area from AOI data if available
            area_hectares = 1.0
            if aoi_data and 'polygon' in aoi_data:
                try:
                    # Try to calculate area from polygon
                    polygon = aoi_data['polygon']
                    if hasattr(polygon, 'area'):
                        # Convert to hectares (rough approximation)
                        area_hectares = abs(polygon.area) * 111319.9 * 111319.9 / 10000  # deg^2 to m^2 to ha
                    elif 'area_ha' in aoi_data:
                        area_hectares = aoi_data['area_ha']
                    print(f"📏 Calculated area: {area_hectares:.1f} hectares")
                except Exception as e:
                    print(f"⚠️  Area calculation failed: {e}")
                    area_hectares = 10.0  # Default area
            
            # Combine with existing results
            enhanced_results = {
                **analysis_results,  # Keep all existing analysis
                'ai_species_recommendations': species_data['species_recommendations'],
                'forest_composition': species_data.get('forest_composition', {}),
                'ecological_context': ecological_context,
                'ai_enhancement': {
                    'methodology': species_data.get('methodology', 'AI-Enhanced Analysis'),
                    'confidence_avg': species_data.get('confidence_avg', 0.75),
                    'total_species_analyzed': species_data.get('total_species_available', 15),
                    'biogeographic_zone': biogeographic_zone,
                    'area_analyzed_ha': area_hectares,
                    'models_used': ['Neural Collaborative Filtering', 'Ecological Knowledge Graph']
                },
                'enhanced_suitability': self._calculate_enhanced_suitability(
                    analysis_results, species_data, ecological_context
                )
            }
            
            print("✅ Analysis enhancement completed successfully")
            return enhanced_results
            
        except Exception as e:
            print(f"⚠️  AI enhancement failed: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            return self._fallback_recommendations(analysis_results, coordinates)
    
    def _calculate_enhanced_suitability(self, analysis_results: Dict, 
                                      species_data: Dict, 
                                      ecological_context: Dict) -> Dict:
        """Calculate enhanced suitability score incorporating AI insights"""
        
        # Get original suitability if available
        original_suitability = analysis_results.get('suitability', {})
        original_score = original_suitability.get('composite_score', 0.7)
        
        # Get AI confidence average
        ai_confidence = species_data.get('confidence_avg', 0.7)
        
        # Get ecological restoration potential
        ecological_potential = 0.7  # Default
        if ecological_context.get('restoration_potential'):
            ecological_potential = ecological_context['restoration_potential'].get('potential_score', 0.7)
        
        # Calculate enhanced score (weighted combination)
        enhanced_score = (
            original_score * 0.4 +      # Original environmental analysis
            ai_confidence * 0.4 +       # AI species recommendations confidence
            ecological_potential * 0.2  # Ecological context
        )
        
        # Determine enhanced grade
        if enhanced_score >= 0.85:
            grade = 'A+'
            description = 'Exceptional restoration potential'
        elif enhanced_score >= 0.75:
            grade = 'A'
            description = 'Excellent restoration potential'
        elif enhanced_score >= 0.65:
            grade = 'B+'
            description = 'Very good restoration potential'
        elif enhanced_score >= 0.55:
            grade = 'B'
            description = 'Good restoration potential'
        elif enhanced_score >= 0.45:
            grade = 'C+'
            description = 'Fair restoration potential'
        elif enhanced_score >= 0.35:
            grade = 'C'
            description = 'Moderate restoration potential'
        else:
            grade = 'D'
            description = 'Challenging restoration conditions'
        
        return {
            'enhanced_score': round(enhanced_score, 3),
            'enhanced_grade': grade,
            'description': description,
            'improvement_over_original': round(enhanced_score - original_score, 3),
            'ai_contribution': round(ai_confidence * 0.4, 3),
            'ecological_contribution': round(ecological_potential * 0.2, 3),
            'confidence_level': 'High' if enhanced_score > 0.7 else 'Medium' if enhanced_score > 0.5 else 'Low'
        }
    
    def _fallback_recommendations(self, analysis_results: Dict, 
                                coordinates: Optional[Tuple] = None) -> Dict:
        """Fallback recommendations if AI models fail"""
        
        print("🔄 Using fallback recommendations (rule-based)")
        
        # Extract basic environmental data
        ndvi_data = analysis_results.get('ndvi_data', analysis_results.get('ndvi', {}))
        rain_data = analysis_results.get('environmental_data', analysis_results.get('rainfall', {}))
        soil_data = analysis_results.get('soil_data', analysis_results.get('soil', {}))
        
        rainfall = rain_data.get('annual_rainfall', 1000)
        ndvi = ndvi_data.get('ndvi_mean', 0.5)
        soil_ph = soil_data.get('soil_ph', 6.5)
        
        # Simple rule-based recommendations
        if rainfall > 1500:
            primary_species = "Sal"
            scientific_name = "Shorea robusta"
            forest_type = "Tropical Moist Forest"
            confidence = 78
        elif rainfall > 800:
            primary_species = "Teak"
            scientific_name = "Tectona grandis"
            forest_type = "Tropical Deciduous Forest"
            confidence = 72
        else:
            primary_species = "Neem"
            scientific_name = "Azadirachta indica"
            forest_type = "Dry Deciduous Forest"
            confidence = 68
        
        # Additional species based on conditions
        fallback_recommendations = [
            {
                "name": primary_species,
                "scientific": scientific_name,
                "family": "Unknown",
                "layer": "Canopy",
                "score": confidence / 100,
                "confidence": float(confidence),
                "carbon": 200,
                "growth": "Medium",
                "features": ["Native Species", "Adaptive"],
                "rainfall": f"{max(400, rainfall-200)}-{rainfall+500}",
                "ecological_context": f"Rule-based selection for {rainfall}mm rainfall region"
            }
        ]
        
        # Add secondary species
        secondary_species = [
            ("Bamboo", "Dendrocalamus strictus", confidence - 10),
            ("Jamun", "Syzygium cumini", confidence - 15),
            ("Amla", "Phyllanthus emblica", confidence - 20)
        ]
        
        for name, scientific, conf in secondary_species:
            fallback_recommendations.append({
                "name": name,
                "scientific": scientific,
                "family": "Unknown",
                "layer": "Understory",
                "score": conf / 100,
                "confidence": float(conf),
                "carbon": 150,
                "growth": "Fast",
                "features": ["Native", "Hardy"],
                "rainfall": f"{rainfall-300}-{rainfall+300}",
                "ecological_context": f"Complementary species for mixed forest"
            })
        
        # Determine biogeographic zone (simplified)
        biogeographic_zone = 'Unknown'
        if coordinates:
            lat, lon = coordinates
            if 8 <= lat <= 21 and 72 <= lon <= 77:
                biogeographic_zone = 'Western_Ghats'
            elif 22 <= lat <= 30 and 75 <= lon <= 89:
                biogeographic_zone = 'Gangetic_Plain'
            else:
                biogeographic_zone = 'Deccan_Peninsula'
        
        return {
            **analysis_results,
            'ai_species_recommendations': fallback_recommendations,
            'ecological_context': {
                'biogeographic_zone': biogeographic_zone,
                'forest_type': forest_type,
                'restoration_approach': f'Mixed native forest for {biogeographic_zone}'
            },
            'ai_enhancement': {
                'methodology': 'Rule-based fallback system',
                'confidence_avg': confidence / 100,
                'total_species_analyzed': len(fallback_recommendations),
                'biogeographic_zone': biogeographic_zone,
                'models_used': ['Environmental Decision Trees'],
                'note': 'AI models not available, using simplified recommendations'
            },
            'enhanced_suitability': {
                'enhanced_score': 0.65,
                'enhanced_grade': 'B',
                'description': 'Good potential with basic analysis',
                'confidence_level': 'Medium'
            }
        }

# Main function to integrate with Streamlit
def get_enhanced_manthan_results(analysis_results: Dict, 
                               aoi_data: Optional[Dict] = None,
                               coordinates: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Main function to replace mock data in your Streamlit app
    
    Usage in Streamlit:
    ```python
    # Replace existing species_data with:
    enhanced_results = get_enhanced_manthan_results(
        st.session_state["analysis_results"],
        st.session_state.get("aoi_data")
    )
    species_data = enhanced_results['ai_species_recommendations']
    ```
    """
    
    print("🚀 Starting Manthan AI enhancement...")
    
    # Extract coordinates from AOI if available and not provided
    if not coordinates and aoi_data and 'polygon' in aoi_data:
        try:
            polygon = aoi_data['polygon']
            bounds = polygon.bounds
            # Get center coordinates
            coordinates = (
                (bounds[1] + bounds[3]) / 2,  # latitude
                (bounds[0] + bounds[2]) / 2   # longitude
            )
            print(f"📍 Extracted coordinates from AOI: {coordinates}")
        except Exception as e:
            print(f"⚠️  Could not extract coordinates: {e}")
    
    # Initialize integration
    ai_integration = ManthanAIIntegration()
    
    # Get enhanced results
    enhanced_results = ai_integration.enhance_analysis_results(
        analysis_results, coordinates, aoi_data
    )
    
    print("✅ Manthan AI enhancement completed")
    return enhanced_results

# Utility functions for specific Streamlit components
def get_species_for_streamlit_display(enhanced_results: Dict) -> List[Dict]:
    """Format species data for Streamlit display - matches your existing format"""
    
    species_list = enhanced_results.get('ai_species_recommendations', [])
    
    # Format for your existing Streamlit species display
    formatted_species = []
    for i, species in enumerate(species_list[:8]):  # Top 8 for display
        formatted_species.append({
            "name": species.get('name', 'Unknown'),
            "scientific": species.get('scientific', 'Unknown'),
            "family": species.get('family', 'Unknown'),
            "layer": species.get('layer', 'Canopy'),
            "growth": species.get('growth', species.get('growth_rate', 'Medium')),
            "carbon": species.get('carbon', species.get('carbon_sequestration', 200)),
            "rainfall": species.get('rainfall', species.get('rainfall_range', '800-2000 mm')),
            "features": species.get('features', ['Native Species']),
            "score": int(species.get('confidence', species.get('score', 0.75) * 100))
        })
    
    return formatted_species

def get_ecological_summary_for_streamlit(enhanced_results: Dict) -> Dict:
    """Get ecological context summary for Streamlit display"""
    
    ecological_context = enhanced_results.get('ecological_context', {})
    ai_enhancement = enhanced_results.get('ai_enhancement', {})
    enhanced_suitability = enhanced_results.get('enhanced_suitability', {})
    
    return {
        'biogeographic_zone': ecological_context.get('biogeographic_zone', 
                                                   ai_enhancement.get('biogeographic_zone', 'Unknown')),
        'forest_type': ecological_context.get('restoration_recommendations', {}).get('primary_approach', 'Mixed Forest'),
        'success_probability': ecological_context.get('restoration_potential', {}).get('success_probability', 75),
        'methodology': ai_enhancement.get('methodology', 'Standard Analysis'),
        'confidence_avg': ai_enhancement.get('confidence_avg', 0.7),
        'enhanced_grade': enhanced_suitability.get('enhanced_grade', 'B'),
        'enhanced_score': enhanced_suitability.get('enhanced_score', 0.7),
        'models_used': ai_enhancement.get('models_used', ['Basic Analysis']),
        'area_analyzed': ai_enhancement.get('area_analyzed_ha', 1.0),
        'total_species': ai_enhancement.get('total_species_analyzed', 0)
    }

def get_forest_composition_for_streamlit(enhanced_results: Dict) -> Dict:
    """Get forest composition data formatted for Streamlit charts"""
    
    composition = enhanced_results.get('forest_composition', {})
    
    if not composition:
        # Return default composition
        return {
            'layers': ['Emergent', 'Canopy', 'Understory', 'Ground'],
            'percentages': [15, 45, 30, 10],
            'tree_counts': [375, 1125, 750, 250],
            'primary_species': ['Mixed', 'Mixed', 'Mixed', 'Mixed']
        }
    
    layers = []
    percentages = []
    tree_counts = []
    primary_species = []
    
    for layer_name, layer_data in composition.items():
        if isinstance(layer_data, dict) and 'percentage' in layer_data:
            layers.append(layer_name)
            percentages.append(layer_data['percentage'])
            tree_counts.append(layer_data['trees_count'])
            primary_species.append(layer_data['primary_species'])
    
    return {
        'layers': layers,
        'percentages': percentages,
        'tree_counts': tree_counts,
        'primary_species': primary_species,
        'total_trees': composition.get('summary', {}).get('total_trees', 2500),
        'carbon_potential': composition.get('summary', {}).get('estimated_carbon_sequestration', 500)
    }

# Integration helper for replacing mock data in existing Streamlit
def integrate_with_existing_streamlit_species_section(enhanced_results: Dict) -> str:
    """
    Generate code snippet to replace mock data in Streamlit species section
    """
    
    species_data = get_species_for_streamlit_display(enhanced_results)
    ecological_summary = get_ecological_summary_for_streamlit(enhanced_results)
    
    integration_code = f'''
# Replace your existing species_data list with this AI-powered version:

# AI Enhancement Information
if st.session_state.get("analysis_results"):
    enhanced_results = get_enhanced_manthan_results(
        st.session_state["analysis_results"],
        st.session_state.get("aoi_data")
    )
    
    # Get AI-powered species recommendations
    species_data = get_species_for_streamlit_display(enhanced_results)
    ecological_summary = get_ecological_summary_for_streamlit(enhanced_results)
    
    # Display AI context
    st.info(f"🧠 AI Analysis: {{ecological_summary['biogeographic_zone']}} region detected. "
            f"Enhanced suitability grade: {{ecological_summary['enhanced_grade']}} "
            f"({{ecological_summary['enhanced_score']:.1%}})")
    
    # Show methodology
    st.caption(f"Methodology: {{ecological_summary['methodology']}} | "
               f"{{ecological_summary['total_species']}} species analyzed")
else:
    # Your existing fallback code here
    species_data = [
        # ... your existing mock data ...
    ]
'''
    
    return integration_code

# Test function
def test_integration():
    """Test the integration with sample data"""
    
    print("🧪 Testing Manthan AI Integration...")
    
    # Sample analysis results (like from your day2_pipeline)
    sample_results = {
        'ndvi': {'ndvi_mean': 0.65, 'ndvi_std': 0.12},
        'rainfall': {'annual_rainfall': 1200},
        'soil': {'soil_ph': 6.8},
        'suitability': {'composite_score': 0.78, 'suitability_grade': 'B+'}
    }
    
    # Test coordinates (Bangalore area)
    test_coordinates = (12.9716, 77.5946)
    
    # Test the enhancement
    try:
        enhanced = get_enhanced_manthan_results(sample_results, coordinates=test_coordinates)
        
        print("✅ Integration test successful!")
        print(f"🌿 AI Species Recommendations: {len(enhanced['ai_species_recommendations'])}")
        print(f"🌍 Biogeographic Zone: {enhanced['ai_enhancement']['biogeographic_zone']}")
        print(f"📊 Enhanced Grade: {enhanced['enhanced_suitability']['enhanced_grade']}")
        
        # Test Streamlit formatting
        species_display = get_species_for_streamlit_display(enhanced)
        print(f"📋 Formatted for Streamlit: {len(species_display)} species")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_integration()

def get_forest_composition_for_streamlit(enhanced_results: Dict) -> Dict:
    """Get forest composition data formatted for Streamlit charts"""
    
    composition = enhanced_results.get('forest_composition', {})
    
    if not composition:
        # Return default composition for desert/arid zones
        return {
            'layers': ['Canopy', 'Understory', 'Shrub', 'Ground Cover'],
            'percentages': [25, 35, 30, 10],
            'tree_counts': [625, 875, 750, 250],
            'primary_species': ['Khejri', 'Neem', 'Babul', 'Grasses']
        }
    
    layers = []
    percentages = []
    tree_counts = []
    primary_species = []
    
    for layer_name, layer_data in composition.items():
        if isinstance(layer_data, dict) and 'percentage' in layer_data:
            layers.append(layer_name)
            percentages.append(layer_data['percentage'])
            tree_counts.append(layer_data['trees_count'])
            primary_species.append(layer_data['primary_species'])
    
    return {
        'layers': layers if layers else ['Canopy', 'Understory', 'Shrub', 'Ground'],
        'percentages': percentages if percentages else [25, 35, 30, 10],
        'tree_counts': tree_counts if tree_counts else [625, 875, 750, 250],
        'primary_species': primary_species if primary_species else ['Mixed', 'Mixed', 'Mixed', 'Mixed'],
        'total_trees': composition.get('summary', {}).get('total_trees', 2500),
        'carbon_potential': composition.get('summary', {}).get('estimated_carbon_sequestration', 400)
    }
