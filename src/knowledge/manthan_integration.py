"""
Integration module to connect the Knowledge Graph with Manthan's existing pipeline
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .indian_forest_knowledge_graph import IndianForestKnowledgeGraph
from .region_mapper import IndianRegionMapper

class ManthanEcologicalIntelligence:
    """
    Main integration class that brings together all ecological intelligence components
    """
    
    def __init__(self):
        self.knowledge_graph = IndianForestKnowledgeGraph()
        self.region_mapper = IndianRegionMapper()
        self.initialize_caches()
    
    def initialize_caches(self):
        """Initialize caches for performance"""
        self._species_cache = {}
        self._forest_type_cache = {}
        self._recommendation_cache = {}
    
    def analyze_location(self, aoi_geojson: Dict, 
                        environmental_data: Dict = None) -> Dict:
        """
        Complete ecological analysis for a location
        
        Args:
            aoi_geojson: GeoJSON of area of interest
            environmental_data: Optional env data (rainfall, temp, soil_ph)
            
        Returns:
            Comprehensive ecological analysis
        """
        # Extract coordinates
        coords = self._extract_centroid(aoi_geojson)
        lat, lon = coords['lat'], coords['lon']
        
        # Get region metadata
        region_meta = self.region_mapper.get_region_metadata(lat, lon)
        
        # Get environmental data (use provided or defaults)
        env_data = environmental_data or self._get_default_environmental_data(region_meta)
        
        # Determine forest type
        forest_type = self._determine_forest_type(region_meta, env_data)
        
        # Get ecological structure
        structure = self.knowledge_graph.get_ecological_structure(
            forest_type,
            region_meta['biogeographic_zone']
        )
        
        # Get species recommendations
        recommendations = self._get_species_recommendations(
            state=region_meta['state'],
            forest_type=forest_type,
            env_data=env_data,
            area_ha=coords.get('area_ha', 10)
        )
        
        # Compile analysis
        analysis = {
            "location": {
                "coordinates": coords,
                "state": region_meta['state'],
                "biogeographic_zone": region_meta['biogeographic_zone']
            },
            "forest_ecology": {
                "identified_type": forest_type,
                "possible_types": region_meta['forest_types'],
                "reference_sites": self.region_mapper.get_nearby_reference_sites(lat, lon)
            },
            "environmental_conditions": env_data,
            "ecological_structure": structure,
            "species_recommendations": recommendations,
            "implementation_guidance": self._generate_implementation_guidance(
                forest_type, structure, recommendations
            ),
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "confidence_score": self._calculate_confidence_score(region_meta, env_data)
            }
        }
        
        return analysis
    
    def _extract_centroid(self, aoi_geojson: Dict) -> Dict:
        """Extract centroid and area from AOI"""
        # Simplified centroid extraction
        if aoi_geojson.get('geometry', {}).get('type') == 'Polygon':
            coords = aoi_geojson['geometry']['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            return {
                'lat': np.mean(lats),
                'lon': np.mean(lons),
                'area_ha': aoi_geojson.get('properties', {}).get('area_ha', 10)
            }
        
        return {'lat': 20.0, 'lon': 78.0, 'area_ha': 10}  # Default India center
    
    def _get_default_environmental_data(self, region_meta: Dict) -> Dict:
        """Get default environmental data for a region"""
        # Use middle of the range as default
        rainfall_range = region_meta.get('rainfall_range', [800, 1500])
        
        return {
            'annual_rainfall': np.mean(rainfall_range),
            'temperature': 27,  # Default Indian average
            'soil_ph': 6.5,     # Neutral default
            'elevation': np.mean(region_meta.get('elevation_range', [200, 800]))
        }
    
    def _determine_forest_type(self, region_meta: Dict, env_data: Dict) -> str:
        """Determine most suitable forest type based on conditions"""
        possible_types = region_meta['forest_types']
        
        if not possible_types:
            return "Tropical Dry Deciduous"  # Default
        
        # Score each forest type
        scores = {}
        for ftype in possible_types:
            score = 0
            
            # Rainfall matching
            if "Wet" in ftype and env_data['annual_rainfall'] > 2000:
                score += 30
            elif "Moist" in ftype and 1000 < env_data['annual_rainfall'] < 2500:
                score += 30
            elif "Dry" in ftype and env_data['annual_rainfall'] < 1200:
                score += 30
            
            # Elevation matching
            if "Montane" in ftype and env_data.get('elevation', 500) > 1500:
                score += 20
            elif "Hill" in ftype and 800 < env_data.get('elevation', 500) < 2000:
                score += 20
            
            # Temperature matching
            if "Temperate" in ftype and env_data.get('temperature', 27) < 20:
                score += 20
            elif "Tropical" in ftype and env_data.get('temperature', 27) > 20:
                score += 20
            
            scores[ftype] = score
        
        # Return highest scoring type
        return max(scores, key=scores.get)
    
    def _get_species_recommendations(self, state: str, forest_type: str,
                                   env_data: Dict, area_ha: float) -> Dict:
        """Get detailed species recommendations"""
        # Check cache first
        cache_key = f"{state}_{forest_type}_{env_data['annual_rainfall']}"
        if cache_key in self._recommendation_cache:
            return self._recommendation_cache[cache_key]
        
        # Get species list from knowledge graph
        species_by_layer = self.knowledge_graph.get_species_for_forest_type(
            forest_type=forest_type,
            state=state,
            rainfall=env_data['annual_rainfall'],
            soil_ph=env_data['soil_ph']
        )
        
        # Calculate planting numbers
        trees_per_ha = 2500  # Miyawaki-style dense planting
        total_trees = int(area_ha * trees_per_ha)
        
        # Get layer proportions
        structure = self.knowledge_graph.get_ecological_structure(forest_type, state)
        
        recommendations = {
            "summary": {
                "total_trees": total_trees,
                "total_species": sum(len(species) for species in species_by_layer.values()),
                "trees_per_ha": trees_per_ha,
                "area_ha": area_ha
            },
            "by_layer": {}
        }
        
        # Calculate trees per layer
        for layer, proportion in structure['layer_proportions'].items():
            layer_trees = int(total_trees * proportion / 100)
            species_in_layer = species_by_layer.get(layer, [])
            
            if species_in_layer:
                trees_per_species = layer_trees // len(species_in_layer)
                
                recommendations["by_layer"][layer] = {
                    "total_trees": layer_trees,
                    "proportion": proportion,
                    "species": [
                        {
                            "name": sp['common_name'],
                            "scientific_name": sp['scientific_name'],
                            "family": sp['family'],
                            "trees_to_plant": trees_per_species,
                            "carbon_sequestration": sp.get('carbon_sequestration', 0),
                            "ecological_value": sp.get('ecological_value', []),
                            "planting_notes": sp.get('planting_notes', '')
                        }
                        for sp in species_in_layer
                    ]
                }
        
        # Cache the result
        self._recommendation_cache[cache_key] = recommendations
        
        return recommendations
    
    def _generate_implementation_guidance(self, forest_type: str,
                                        structure: Dict,
                                        recommendations: Dict) -> Dict:
        """Generate practical implementation guidance"""
        guidance = {
            "planting_strategy": self._get_planting_strategy(forest_type),
            "seasonal_calendar": self._get_seasonal_calendar(forest_type),
            "site_preparation": self._get_site_preparation(forest_type),
            "maintenance_schedule": self._get_maintenance_schedule(),
            "success_indicators": self._get_success_indicators(forest_type)
        }
        
        return guidance
    
    def _get_planting_strategy(self, forest_type: str) -> Dict:
        """Get planting strategy based on forest type"""
        strategies = {
            "default": {
                "method": "Miyawaki Dense Planting",
                "spacing": "1-1.5m between saplings",
                "pattern": "Random natural pattern",
                "phases": [
                    "Phase 1: Pioneer species and soil builders",
                    "Phase 2: Main canopy species (after 6 months)",
                    "Phase 3: Climax species (after 1 year)"
                ]
            }
        }
        
        # Customize by forest type
        if "Wet" in forest_type:
            strategies["default"]["special_considerations"] = [
                "Ensure good drainage to prevent waterlogging",
                "Plant on mounds in heavy rainfall areas",
                "Include epiphyte-supporting species"
            ]
        elif "Dry" in forest_type:
            strategies["default"]["special_considerations"] = [
                "Deep planting for better water access",
                "Mulching essential for moisture retention",
                "Focus on drought-resistant rootstock"
            ]
        
        return strategies["default"]
    
    def _get_seasonal_calendar(self, forest_type: str) -> List[Dict]:
        """Get seasonal planting calendar"""
        # Default Indian monsoon-based calendar
        calendar = [
            {
                "month": "June-July",
                "activity": "Main planting season",
                "tasks": ["Pit preparation", "Planting of all species", "Initial staking"]
            },
            {
                "month": "August-September", 
                "activity": "Establishment care",
                "tasks": ["Weeding", "Replacement planting", "Growth monitoring"]
            },
            {
                "month": "October-November",
                "activity": "Post-monsoon care",
                "tasks": ["Mulching", "Pruning if needed", "Pest monitoring"]
            },
            {
                "month": "December-February",
                "activity": "Dry season maintenance",
                "tasks": ["Watering schedule", "Fire prevention", "Growth measurement"]
            },
            {
                "month": "March-May",
                "activity": "Pre-monsoon preparation",
                "tasks": ["Site preparation for new areas", "Nursery preparation", "Tool maintenance"]
            }
        ]
        
        return calendar
    
    def _get_site_preparation(self, forest_type: str) -> List[str]:
        """Get site preparation guidelines"""
        guidelines = [
            "Remove invasive species completely",
            "Prepare planting pits (30x30x30 cm minimum)",
            "Mix native soil with compost (70:30 ratio)",
            "Install water harvesting structures",
            "Set up protection fencing",
            "Mark planting spots maintaining natural randomness"
        ]
        
        return guidelines
    
    def _get_maintenance_schedule(self) -> Dict:
        """Get maintenance schedule"""
        return {
            "Year 1": {
                "frequency": "Weekly",
                "activities": ["Watering", "Weeding", "Monitoring survival"]
            },
            "Year 2": {
                "frequency": "Fortnightly", 
                "activities": ["Watering in dry season", "Weeding", "Pruning"]
            },
            "Year 3": {
                "frequency": "Monthly",
                "activities": ["Minimal watering", "Monitoring growth", "Thinning if needed"]
            },
            "Year 4+": {
                "frequency": "Quarterly",
                "activities": ["Health monitoring", "Natural regeneration support", "Documentation"]
            }
        }
    
    def _get_success_indicators(self, forest_type: str) -> List[Dict]:
        """Get success indicators for monitoring"""
        return [
            {
                "parameter": "Survival Rate",
                "target": "85%+",
                "measurement": "Count living vs planted"
            },
            {
                "parameter": "Height Growth",
                "target": "3-5m in 3 years",
                "measurement": "Average height of canopy species"
            },
            {
                "parameter": "Canopy Cover",
                "target": "80%+ by year 5",
                "measurement": "Densiometer or drone imagery"
            },
            {
                "parameter": "Biodiversity",
                "target": "50+ species (including fauna)",
                "measurement": "Species counts, camera traps"
            },
            {
                "parameter": "Soil Health",
                "target": "Increased organic matter",
                "measurement": "Soil tests every 2 years"
            }
        ]
    
    def _calculate_confidence_score(self, region_meta: Dict, env_data: Dict) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.5  # Base score
        
        # Add points for data availability
        if region_meta.get('state'):
            score += 0.1
        if region_meta.get('biogeographic_zone') != "Unknown":
            score += 0.1
        if env_data.get('annual_rainfall'):
            score += 0.1
        if env_data.get('soil_ph'):
            score += 0.1
        if len(region_meta.get('forest_types', [])) > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    # Streamlit UI Integration Methods
    def render_ecological_analysis(self, analysis: Dict):
        """Render ecological analysis in Streamlit UI"""
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">🌍</div>
                <div>
                    <h3 class="card-title">Ecological Context Analysis</h3>
                    <p class="card-subtitle">AI-powered site intelligence</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Location Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("State", analysis['location']['state'])
        with col2:
            st.metric("Biogeographic Zone", analysis['location']['biogeographic_zone'])
        with col3:
            st.metric("Forest Type", analysis['forest_ecology']['identified_type'])
        
        # Ecological Structure
        st.subheader("🌳 Natural Forest Structure")
        
        structure = analysis['ecological_structure']
        
        # Create structure visualization
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Bar(
            x=list(structure['layer_proportions'].keys()),
            y=list(structure['layer_proportions'].values()),
            marker_color=['#064e3b', '#059669', '#10b981', '#34d399', '#86efac']
        )])
        
        fig.update_layout(
            title="Recommended Canopy Layer Distribution",
            xaxis_title="Forest Layer",
            yaxis_title="Percentage (%)",
            showlegend=False,
            plot_bgcolor='white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Species Recommendations by Layer
        st.subheader("🌿 Layer-wise Species Recommendations")
        
        recommendations = analysis['species_recommendations']['by_layer']
        
        for layer, layer_data in recommendations.items():
            with st.expander(f"{layer} Layer - {layer_data['total_trees']} trees ({layer_data['proportion']}%)"):
                for species in layer_data['species']:
                    st.markdown(f"""
                    **{species['name']}** (*{species['scientific_name']}*)
                    - Family: {species['family']}
                    - Trees to plant: {species['trees_to_plant']}
                    - Carbon sequestration: {species['carbon_sequestration']} kg CO₂/year/tree
                    - Ecological value: {', '.join(species['ecological_value'])}
                    """)
        
        # Implementation Guidance
        st.subheader("📋 Implementation Guidance")
        
        guidance = analysis['implementation_guidance']
        
        tab1, tab2, tab3, tab4 = st.tabs(["Planting Strategy", "Seasonal Calendar", 
                                         "Site Preparation", "Maintenance"])
        
        with tab1:
            strategy = guidance['planting_strategy']
            st.write(f"**Method:** {strategy['method']}")
            st.write(f"**Spacing:** {strategy['spacing']}")
            st.write(f"**Pattern:** {strategy['pattern']}")
            st.write("**Phases:**")
            for phase in strategy['phases']:
                st.write(f"- {phase}")
        
        with tab2:
            calendar_df = pd.DataFrame(guidance['seasonal_calendar'])
            st.dataframe(calendar_df, use_container_width=True)
        
        with tab3:
            for guideline in guidance['site_preparation']:
                st.write(f"✓ {guideline}")
        
        with tab4:
            for year, schedule in guidance['maintenance_schedule'].items():
                st.write(f"**{year}** - {schedule['frequency']}")
                for activity in schedule['activities']:
                    st.write(f"  • {activity}")
        
        # Confidence Score
        confidence = analysis['metadata']['confidence_score']
        st.progress(confidence)
        st.caption(f"Analysis Confidence: {confidence*100:.0f}%")
    
    def export_restoration_plan(self, analysis: Dict) -> bytes:
        """Export restoration plan as PDF or detailed JSON"""
        # Implementation for PDF export would go here
        # For now, return formatted JSON
        import json
        return json.dumps(analysis, indent=2).encode('utf-8')
