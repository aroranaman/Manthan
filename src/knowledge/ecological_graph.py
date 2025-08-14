"""
Ecological Knowledge Graph for Indian Forest Ecosystems
Location: src/knowledge/ecological_graph.py
Maps coordinates to biogeographic zones and forest types
"""

import networkx as nx
import json
import math
from typing import Dict, List, Tuple, Optional

class EcologicalKnowledgeGraph:
    """Knowledge graph for Indian forest ecosystems and biogeography"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.biogeographic_zones = {}
        self.forest_types = {}
        self.build_knowledge_graph()
    
    def build_knowledge_graph(self):
        """Build the complete ecological knowledge graph for India"""
        self._create_biogeographic_zones()
        self._create_forest_types()
        self._create_relationships()
    
    def _create_biogeographic_zones(self):
        """Create the 10 biogeographic zones of India"""
        
        zones = {
            'Western_Ghats': {
                'bounds': {'lat': (8, 21), 'lon': (72, 77)},
                'rainfall_range': (1500, 6000),
                'elevation_range': (0, 2700),
                'temperature_range': (15, 35),
                'characteristics': ['High endemism', 'UNESCO Hotspot', 'Monsoon dependent'],
                'dominant_vegetation': ['Tropical Wet Evergreen', 'Semi Evergreen', 'Montane'],
                'biodiversity_index': 0.95,
                'restoration_priority': 'Critical'
            },
            
            'Deccan_Peninsula': {
                'bounds': {'lat': (10, 25), 'lon': (74, 85)},
                'rainfall_range': (400, 1500),
                'elevation_range': (100, 1000), 
                'temperature_range': (20, 45),
                'characteristics': ['Dry deciduous', 'Scrublands', 'Drought adapted'],
                'dominant_vegetation': ['Tropical Dry Deciduous', 'Thorn Forest'],
                'biodiversity_index': 0.65,
                'restoration_priority': 'High'
            },
            
            'Gangetic_Plain': {
                'bounds': {'lat': (22, 30), 'lon': (75, 89)},
                'rainfall_range': (600, 1500),
                'elevation_range': (50, 300),
                'temperature_range': (15, 45),
                'characteristics': ['Alluvial soil', 'Agricultural landscape', 'Riverine'],
                'dominant_vegetation': ['Tropical Moist Deciduous', 'Riverine Forest'],
                'biodiversity_index': 0.55,
                'restoration_priority': 'Medium'
            },
            
            'Trans_Himalaya': {
                'bounds': {'lat': (30, 37), 'lon': (75, 81)},
                'rainfall_range': (50, 400),
                'elevation_range': (3000, 8000),
                'temperature_range': (-15, 25),
                'characteristics': ['Cold desert', 'Alpine vegetation', 'Snow adapted'],
                'dominant_vegetation': ['Alpine Scrub', 'Cold Desert'],
                'biodiversity_index': 0.35,
                'restoration_priority': 'Specialized'
            },
            
            'Himalaya': {
                'bounds': {'lat': (28, 35), 'lon': (74, 89)},
                'rainfall_range': (800, 3000),
                'elevation_range': (1000, 8000),
                'temperature_range': (-10, 30),
                'characteristics': ['Altitudinal gradients', 'Temperate forests', 'Snow peaks'],
                'dominant_vegetation': ['Temperate Coniferous', 'Alpine', 'Temperate Deciduous'],
                'biodiversity_index': 0.80,
                'restoration_priority': 'High'
            },
            
            'Northeast_India': {
                'bounds': {'lat': (22, 28), 'lon': (89, 97)},
                'rainfall_range': (1500, 4000),
                'elevation_range': (50, 4000),
                'temperature_range': (10, 35),
                'characteristics': ['High rainfall', 'Bamboo forests', 'Tribal areas', 'Biodiversity hotspot'],
                'dominant_vegetation': ['Tropical Wet Evergreen', 'Bamboo', 'Montane'],
                'biodiversity_index': 0.88,
                'restoration_priority': 'Critical'
            },
            
            'Desert': {
                'bounds': {'lat': (24, 30), 'lon': (68, 75)},
                'rainfall_range': (50, 400),
                'elevation_range': (100, 500),
                'temperature_range': (5, 50),
                'characteristics': ['Arid climate', 'Sand dunes', 'Xerophytic vegetation'],
                'dominant_vegetation': ['Desert Scrub', 'Thorn Forest'],
                'biodiversity_index': 0.25,
                'restoration_priority': 'Specialized'
            },
            
            'Semi_Arid': {
                'bounds': {'lat': (20, 28), 'lon': (70, 78)},
                'rainfall_range': (200, 800),
                'elevation_range': (100, 800),
                'temperature_range': (10, 48),
                'characteristics': ['Semi-arid', 'Grasslands', 'Thorny vegetation'],
                'dominant_vegetation': ['Tropical Thorn', 'Grassland'],
                'biodiversity_index': 0.45,
                'restoration_priority': 'Medium'
            },
            
            'Coasts': {
                'bounds': {'lat': (6, 25), 'lon': (68, 94)},  # Broad coastal definition
                'rainfall_range': (800, 3000),
                'elevation_range': (0, 100),
                'temperature_range': (20, 35),
                'characteristics': ['Saline conditions', 'Mangroves', 'Coastal adaptation'],
                'dominant_vegetation': ['Mangrove', 'Coastal Scrub', 'Littoral'],
                'biodiversity_index': 0.70,
                'restoration_priority': 'High'
            },
            
            'Islands': {
                'bounds': {'lat': (6, 14), 'lon': (72, 94)},  # Andaman, Nicobar, Lakshadweep
                'rainfall_range': (1500, 3500),
                'elevation_range': (0, 700),
                'temperature_range': (22, 32),
                'characteristics': ['Island ecology', 'Endemic species', 'Marine influence'],
                'dominant_vegetation': ['Tropical Evergreen', 'Mangrove', 'Coastal'],
                'biodiversity_index': 0.92,
                'restoration_priority': 'Critical'
            }
        }
        
        # Add zones to graph and store data
        for zone_name, zone_data in zones.items():
            self.graph.add_node(zone_name, type='biogeographic_zone', **zone_data)
            self.biogeographic_zones[zone_name] = zone_data
    
    def _create_forest_types(self):
        """Create forest type classifications based on Champion & Seth (1968)"""
        
        forest_types = {
            'Tropical_Wet_Evergreen': {
                'code': '1A',
                'rainfall_requirement': (2000, 6000),
                'temperature_range': (20, 30),
                'elevation_range': (0, 1000),
                'characteristics': ['Dense canopy', 'High diversity', 'No dry season', 'Multi-layered'],
                'indicator_species': ['Dipterocarpus', 'Hopea', 'Mesua', 'Artocarpus'],
                'carbon_potential': 450,  # tons/ha over 30 years
                'establishment_difficulty': 'High',
                'biodiversity_value': 'Exceptional'
            },
            
            'Tropical_Semi_Evergreen': {
                'code': '2A',
                'rainfall_requirement': (1200, 2500),
                'temperature_range': (18, 32),
                'elevation_range': (0, 1200),
                'characteristics': ['Mixed deciduous-evergreen', 'Moderate diversity', 'Short dry season'],
                'indicator_species': ['Tectona', 'Terminalia', 'Xylia', 'Lagerstroemia'],
                'carbon_potential': 380,
                'establishment_difficulty': 'Medium',
                'biodiversity_value': 'High'
            },
            
            'Tropical_Moist_Deciduous': {
                'code': '3A',
                'rainfall_requirement': (800, 1600),
                'temperature_range': (20, 40),
                'elevation_range': (0, 1000),
                'characteristics': ['Shed leaves in dry season', 'Sal dominant', 'Fire tolerant'],
                'indicator_species': ['Shorea robusta', 'Terminalia', 'Adina', 'Syzygium'],
                'carbon_potential': 320,
                'establishment_difficulty': 'Medium',
                'biodiversity_value': 'High'
            },
            
            'Tropical_Dry_Deciduous': {
                'code': '5A',
                'rainfall_requirement': (500, 1200),
                'temperature_range': (20, 45),
                'elevation_range': (0, 900),
                'characteristics': ['Long dry season', 'Open canopy', 'Fire adapted', 'Drought tolerant'],
                'indicator_species': ['Tectona', 'Anogeissus', 'Boswellia', 'Butea'],
                'carbon_potential': 250,
                'establishment_difficulty': 'Low',
                'biodiversity_value': 'Medium'
            },
            
            'Tropical_Thorn': {
                'code': '6A',
                'rainfall_requirement': (200, 800),
                'temperature_range': (20, 48),
                'elevation_range': (0, 600),
                'characteristics': ['Thorny species', 'Drought adapted', 'Open woodland', 'Xerophytic'],
                'indicator_species': ['Acacia', 'Prosopis', 'Capparis', 'Commiphora'],
                'carbon_potential': 120,
                'establishment_difficulty': 'Low',
                'biodiversity_value': 'Medium'
            },
            
            'Temperate_Coniferous': {
                'code': '12A',
                'rainfall_requirement': (800, 2000),
                'temperature_range': (-5, 25),
                'elevation_range': (1500, 4000),
                'characteristics': ['Coniferous dominance', 'Cold adapted', 'Snow tolerant'],
                'indicator_species': ['Pinus', 'Cedrus', 'Abies', 'Picea'],
                'carbon_potential': 400,
                'establishment_difficulty': 'High',
                'biodiversity_value': 'Medium'
            },
            
            'Mangrove': {
                'code': '4E',
                'rainfall_requirement': (800, 3000),
                'temperature_range': (20, 35),
                'elevation_range': (0, 5),
                'characteristics': ['Salt tolerant', 'Tidal influence', 'Specialized root systems'],
                'indicator_species': ['Rhizophora', 'Avicennia', 'Sonneratia', 'Bruguiera'],
                'carbon_potential': 300,
                'establishment_difficulty': 'Very High',
                'biodiversity_value': 'High'
            },
            
            'Bamboo_Forest': {
                'code': '8E',
                'rainfall_requirement': (1000, 2500),
                'temperature_range': (15, 35),
                'elevation_range': (0, 2000),
                'characteristics': ['Bamboo dominated', 'Fast growing', 'Monoculture tendency'],
                'indicator_species': ['Dendrocalamus', 'Bambusa', 'Melocanna'],
                'carbon_potential': 200,
                'establishment_difficulty': 'Low',
                'biodiversity_value': 'Medium'
            }
        }
        
        # Add forest types to graph
        for forest_name, forest_data in forest_types.items():
            self.graph.add_node(forest_name, type='forest_type', **forest_data)
            self.forest_types[forest_name] = forest_data
    
    def _create_relationships(self):
        """Create relationships between zones and forest types"""
        
        # Define which forest types occur in which zones (based on ecological reality)
        zone_forest_mapping = {
            'Western_Ghats': [
                ('Tropical_Wet_Evergreen', 0.9),
                ('Tropical_Semi_Evergreen', 0.8),
                ('Tropical_Moist_Deciduous', 0.6),
                ('Mangrove', 0.7)
            ],
            'Deccan_Peninsula': [
                ('Tropical_Dry_Deciduous', 0.9),
                ('Tropical_Thorn', 0.8),
                ('Tropical_Semi_Evergreen', 0.4)
            ],
            'Gangetic_Plain': [
                ('Tropical_Moist_Deciduous', 0.8),
                ('Tropical_Semi_Evergreen', 0.6),
                ('Bamboo_Forest', 0.7)
            ],
            'Northeast_India': [
                ('Tropical_Wet_Evergreen', 0.95),
                ('Bamboo_Forest', 0.9),
                ('Tropical_Semi_Evergreen', 0.8),
                ('Temperate_Coniferous', 0.7)
            ],
            'Himalaya': [
                ('Temperate_Coniferous', 0.9),
                ('Tropical_Moist_Deciduous', 0.6)
            ],
            'Trans_Himalaya': [
                ('Temperate_Coniferous', 0.5)
            ],
            'Semi_Arid': [
                ('Tropical_Thorn', 0.9),
                ('Tropical_Dry_Deciduous', 0.6)
            ],
            'Desert': [
                ('Tropical_Thorn', 0.7)
            ],
            'Coasts': [
                ('Mangrove', 0.9),
                ('Tropical_Semi_Evergreen', 0.6)
            ],
            'Islands': [
                ('Tropical_Wet_Evergreen', 0.9),
                ('Mangrove', 0.8)
            ]
        }
        
        # Add edges between zones and forest types with suitability weights
        for zone, forest_types in zone_forest_mapping.items():
            for forest_type, suitability in forest_types:
                if self.graph.has_node(forest_type):
                    self.graph.add_edge(zone, forest_type, 
                                      relationship='contains',
                                      suitability=suitability)
    
    def map_coordinates_to_zone(self, latitude: float, longitude: float) -> str:
        """Map geographical coordinates to biogeographic zone with improved accuracy"""
        
        best_match = None
        best_score = 0
        
        # Check each zone's boundaries with priority scoring
        for zone_name, zone_data in self.biogeographic_zones.items():
            bounds = zone_data['bounds']
            lat_min, lat_max = bounds['lat']
            lon_min, lon_max = bounds['lon']
            
            # Basic boundary check
            if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
                score = 1.0
                
                # Special handling for overlapping/priority zones
                if zone_name == 'Western_Ghats':
                    # Western Ghats has high priority due to specificity
                    if 10 <= latitude <= 20 and 73 <= longitude <= 77:
                        score = 1.5
                elif zone_name == 'Northeast_India':
                    # Northeast has high priority when in range
                    if 23 <= latitude <= 27 and 90 <= longitude <= 95:
                        score = 1.4
                elif zone_name == 'Himalaya':
                    # Himalaya priority at higher latitudes
                    if latitude > 30:
                        score = 1.3
                elif zone_name == 'Islands':
                    # Specific island coordinates
                    if ((6 <= latitude <= 14 and 92 <= longitude <= 94) or  # Andaman & Nicobar
                        (10 <= latitude <= 12 and 72 <= longitude <= 74)):   # Lakshadweep
                        score = 2.0
                    else:
                        score = 0  # Not actually islands
                elif zone_name == 'Coasts':
                    # Coastal areas have lower priority unless very close to coast
                    # This would need actual coastal distance calculation
                    score = 0.8
                elif zone_name == 'Desert':
                    # Thar desert specificity
                    if 25 <= latitude <= 28 and 69 <= longitude <= 73:
                        score = 1.2
                
                if score > best_score:
                    best_score = score
                    best_match = zone_name
        
        # Return best match or default
        return best_match if best_match else 'Deccan_Peninsula'
    
    def get_suitable_forest_types(self, zone: str, environmental_conditions: Dict) -> List[Dict]:
        """Get suitable forest types for zone and environmental conditions"""
        
        zone_data = self.biogeographic_zones.get(zone, {})
        suitable_types = []
        
        rainfall = environmental_conditions.get('rainfall', 1000)
        temperature = environmental_conditions.get('temperature', 27)
        elevation = environmental_conditions.get('elevation', 300)
        
        # Get forest types connected to this zone
        connected_forests = []
        if self.graph.has_node(zone):
            for neighbor in self.graph.neighbors(zone):
                if self.graph.nodes[neighbor]['type'] == 'forest_type':
                    edge_data = self.graph.get_edge_data(zone, neighbor)
                    base_suitability = edge_data.get('suitability', 0.5)
                    connected_forests.append((neighbor, base_suitability))
        
        # Evaluate each connected forest type
        for forest_name, base_suitability in connected_forests:
            forest_data = self.forest_types[forest_name]
            
            # Check environmental suitability
            rain_min, rain_max = forest_data['rainfall_requirement']
            temp_min, temp_max = forest_data['temperature_range']
            elev_min, elev_max = forest_data['elevation_range']
            
            # Calculate individual suitability scores
            rain_score = self._calculate_parameter_suitability(rainfall, rain_min, rain_max, tolerance=500)
            temp_score = self._calculate_parameter_suitability(temperature, temp_min, temp_max, tolerance=5)
            elev_score = self._calculate_parameter_suitability(elevation, elev_min, elev_max, tolerance=200)
            
            # Combine scores
            environmental_score = (rain_score * 0.5 + temp_score * 0.3 + elev_score * 0.2)
            final_suitability = base_suitability * environmental_score
            
            # Only include if reasonably suitable
            if final_suitability >= 0.3:
                suitable_types.append({
                    'forest_type': forest_name,
                    'forest_code': forest_data.get('code', ''),
                    'suitability_score': round(final_suitability, 3),
                    'base_zone_suitability': base_suitability,
                    'environmental_match': round(environmental_score, 3),
                    'characteristics': forest_data['characteristics'],
                    'carbon_potential': forest_data['carbon_potential'],
                    'establishment_difficulty': forest_data['establishment_difficulty'],
                    'biodiversity_value': forest_data['biodiversity_value'],
                    'indicator_species': forest_data.get('indicator_species', []),
                    'climate_requirements': {
                        'rainfall': f"{rain_min}-{rain_max} mm",
                        'temperature': f"{temp_min}-{temp_max}°C",
                        'elevation': f"{elev_min}-{elev_max} m"
                    }
                })
        
        # Sort by suitability
        suitable_types.sort(key=lambda x: x['suitability_score'], reverse=True)
        return suitable_types
    
    def _calculate_parameter_suitability(self, value: float, min_val: float, max_val: float, tolerance: float = 0) -> float:
        """Calculate suitability score for a parameter"""
        if min_val <= value <= max_val:
            return 1.0
        
        # Calculate distance from acceptable range
        if value < min_val:
            distance = min_val - value
        else:
            distance = value - max_val
        
        # Apply tolerance
        if distance <= tolerance:
            return max(0.5, 1.0 - (distance / tolerance) * 0.5)
        else:
            return max(0.1, 1.0 - (distance / (max_val - min_val + tolerance)))
    
    def get_zone_summary(self, zone: str) -> Dict:
        """Get comprehensive summary of a biogeographic zone"""
        
        zone_data = self.biogeographic_zones.get(zone, {})
        
        if not zone_data:
            return {'error': f'Zone {zone} not found'}
        
        # Get connected forest types with their suitability
        forest_types = []
        if self.graph.has_node(zone):
            for neighbor in self.graph.neighbors(zone):
                if self.graph.nodes[neighbor]['type'] == 'forest_type':
                    edge_data = self.graph.get_edge_data(zone, neighbor)
                    suitability = edge_data.get('suitability', 0.5)
                    forest_types.append({
                        'name': neighbor,
                        'suitability': suitability,
                        'code': self.forest_types[neighbor].get('code', '')
                    })
        
        # Sort by suitability
        forest_types.sort(key=lambda x: x['suitability'], reverse=True)
        
        return {
            'zone_name': zone,
            'characteristics': zone_data['characteristics'],
            'climate_range': {
                'rainfall': f"{zone_data['rainfall_range'][0]}-{zone_data['rainfall_range'][1]} mm",
                'temperature': f"{zone_data['temperature_range'][0]}-{zone_data['temperature_range'][1]}°C",
                'elevation': f"{zone_data['elevation_range'][0]}-{zone_data['elevation_range'][1]} m"
            },
            'biodiversity_index': zone_data['biodiversity_index'],
            'restoration_priority': zone_data['restoration_priority'],
            'dominant_vegetation': zone_data['dominant_vegetation'],
            'suitable_forest_types': forest_types,
            'conservation_status': self._get_conservation_status(zone_data['biodiversity_index']),
            'restoration_recommendations': self._get_zone_restoration_advice(zone, zone_data)
        }
    
    def _get_conservation_status(self, biodiversity_index: float) -> str:
        """Get conservation status based on biodiversity index"""
        if biodiversity_index >= 0.8:
            return "Critical Biodiversity Area"
        elif biodiversity_index >= 0.6:
            return "High Biodiversity Area"
        elif biodiversity_index >= 0.4:
            return "Medium Biodiversity Area"
        else:
            return "Low Biodiversity Area"
    
    def _get_zone_restoration_advice(self, zone: str, zone_data: Dict) -> Dict:
        """Get restoration advice specific to biogeographic zone"""
        
        biodiversity_index = zone_data['biodiversity_index']
        rainfall_min, rainfall_max = zone_data['rainfall_range']
        
        if zone == 'Western_Ghats':
            approach = "Multi-layered forest restoration with endemic species focus"
            success_factors = ["Monsoon timing", "Slope stabilization", "Endemic species conservation"]
        elif zone == 'Northeast_India':
            approach = "Bamboo-integrated forest restoration with community participation"
            success_factors = ["High rainfall management", "Bamboo integration", "Community involvement"]
        elif zone == 'Desert' or zone == 'Semi_Arid':
            approach = "Drought-resistant species with water conservation"
            success_factors = ["Water harvesting", "Drought-resistant species", "Gradual establishment"]
        elif zone == 'Himalaya':
            approach = "Altitude-specific species selection with erosion control"
            success_factors = ["Altitude adaptation", "Cold tolerance", "Erosion control"]
        elif zone == 'Coasts':
            approach = "Salt-tolerant species with mangrove integration"
            success_factors = ["Salt tolerance", "Tidal adaptation", "Coastal protection"]
        else:
            approach = "Mixed native forest with adaptive management"
            success_factors = ["Site preparation", "Native species", "Adaptive management"]
        
        return {
            'restoration_approach': approach,
            'success_factors': success_factors,
            'establishment_timeline': f"{3 if biodiversity_index > 0.7 else 5}-7 years",
            'maintenance_period': f"{2 if rainfall_min > 1000 else 4} years intensive care",
            'success_probability': int(60 + biodiversity_index * 30)
        }
    
    def analyze_restoration_site(self, latitude: float, longitude: float, 
                               environmental_conditions: Dict) -> Dict:
        """Complete site analysis for restoration planning"""
        
        # Map to biogeographic zone
        zone = self.map_coordinates_to_zone(latitude, longitude)
        
        # Get zone summary
        zone_summary = self.get_zone_summary(zone)
        
        # Get suitable forest types
        suitable_forests = self.get_suitable_forest_types(zone, environmental_conditions)
        
        # Calculate overall restoration potential
        restoration_potential = self._calculate_restoration_potential(
            zone_summary, suitable_forests, environmental_conditions
        )
        
        return {
            'coordinates': {'latitude': latitude, 'longitude': longitude},
            'biogeographic_zone': zone,
            'zone_characteristics': zone_summary,
            'suitable_forest_types': suitable_forests,
            'restoration_assessment': restoration_potential,
            'recommendations': self._generate_restoration_recommendations(
                zone, suitable_forests, environmental_conditions
            )
        }
    
    def _calculate_restoration_potential(self, zone_summary: Dict, 
                                       suitable_forests: List[Dict], 
                                       env_conditions: Dict) -> Dict:
        """Calculate overall restoration potential"""
        
        if not suitable_forests:
            return {
                'potential_score': 0.3,
                'potential_grade': 'Low',
                'limiting_factors': ['No suitable forest types identified'],
                'success_probability': 40
            }
        
        # Base score from best forest type
        best_forest_score = suitable_forests[0]['suitability_score']
        
        # Zone biodiversity bonus
        biodiversity_bonus = zone_summary.get('biodiversity_index', 0.5) * 0.2
        
        # Environmental stability (rainfall consistency indicator)
        rainfall = env_conditions.get('rainfall', 1000)
        if 800 <= rainfall <= 2000:  # Ideal range
            rainfall_stability = 0.1
        else:
            rainfall_stability = 0.05
        
        # Calculate final score
        final_score = min(1.0, best_forest_score + biodiversity_bonus + rainfall_stability)
        
        # Determine grade
        if final_score >= 0.8:
            grade = 'Excellent'
        elif final_score >= 0.6:
            grade = 'Good'
        elif final_score >= 0.4:
            grade = 'Fair'
        else:
            grade = 'Poor'
        
        # Identify limiting factors
        limiting_factors = []
        if best_forest_score < 0.6:
            limiting_factors.append('Suboptimal environmental conditions')
        if zone_summary.get('biodiversity_index', 0) < 0.5:
            limiting_factors.append('Low baseline biodiversity')
        if rainfall < 400 or rainfall > 4000:
            limiting_factors.append('Extreme rainfall conditions')
        
        return {
            'potential_score': round(final_score, 3),
            'potential_grade': grade,
            'success_probability': int(final_score * 90 + 10),  # 10-100% range
            'limiting_factors': limiting_factors,
            'enhancement_opportunities': self._identify_enhancement_opportunities(
                final_score, suitable_forests, env_conditions
            )
        }
    
    def _identify_enhancement_opportunities(self, score: float, 
                                          suitable_forests: List[Dict],
                                          env_conditions: Dict) -> List[str]:
        """Identify opportunities to enhance restoration success"""
        
        opportunities = []
        
        if score < 0.6:
            opportunities.append("Site preparation and soil amendment")
            opportunities.append("Water conservation infrastructure")
        
        if len(suitable_forests) <= 2:
            opportunities.append("Expand species palette with climate-adapted varieties")
        
        rainfall = env_conditions.get('rainfall', 1000)
        if rainfall < 800:
            opportunities.append("Drip irrigation system for establishment phase")
        elif rainfall > 2500:
            opportunities.append("Drainage management for excess water")
        
        opportunities.append("Community engagement and participatory monitoring")
        
        return opportunities
    
    def _generate_restoration_recommendations(self, zone: str, 
                                            suitable_forests: List[Dict],
                                            env_conditions: Dict) -> Dict:
        """Generate specific restoration recommendations"""
        
        if not suitable_forests:
            return {
                'primary_approach': 'Site amelioration required',
                'species_strategy': 'Pioneer species establishment',
                'timeline': 'Extended (7-10 years)',
                'special_considerations': ['Site preparation critical', 'Intensive monitoring required']
            }
        
        primary_forest = suitable_forests[0]
        
        recommendations = {
            'primary_approach': f"Establish {primary_forest['forest_type'].replace('_', ' ')} ecosystem",
            'species_strategy': f"Focus on {', '.join(primary_forest['indicator_species'][:3])} as keystone species",
            'establishment_timeline': f"{3 if primary_forest['establishment_difficulty'] == 'Low' else 5}-7 years",
            'carbon_sequestration_potential': f"{primary_forest['carbon_potential']} tons/ha over 30 years",
            'biodiversity_value': primary_forest['biodiversity_value'],
            'maintenance_requirements': self._get_maintenance_recommendations(primary_forest, env_conditions),
            'monitoring_priorities': [
                'Species establishment success',
                'Soil health improvement', 
                'Biodiversity recovery',
                'Carbon accumulation'
            ]
        }
        
        return recommendations
    
    def _get_maintenance_recommendations(self, forest_type: Dict, env_conditions: Dict) -> List[str]:
        """Get maintenance recommendations based on forest type and conditions"""
        
        maintenance = []
        
        difficulty = forest_type['establishment_difficulty']
        rainfall = env_conditions.get('rainfall', 1000)
        
        if difficulty == 'High':
            maintenance.extend([
                "Intensive watering for first 3 years",
                "Regular weeding and pest monitoring",
                "Soil amendment and mulching"
            ])
        elif difficulty == 'Medium':
            maintenance.extend([
                "Supplemental watering during dry seasons",
                "Annual weeding and maintenance"
            ])
        else:  # Low difficulty
            maintenance.extend([
                "Minimal intervention required",
                "Annual monitoring visits"
            ])
        
        if rainfall < 800:
            maintenance.append("Drought stress monitoring and mitigation")
        elif rainfall > 2500:
            maintenance.append("Drainage management and fungal disease prevention")
        
        return maintenance


# Utility function for integration with Streamlit
def get_ecological_context(latitude: float, longitude: float, 
                          environmental_conditions: Dict) -> Dict:
    """
    Get ecological context for coordinates - integrates with Streamlit
    
    Args:
        latitude, longitude: GPS coordinates
        environmental_conditions: Dict with rainfall, temperature, etc.
        
    Returns:
        Ecological context including zone, forest types, and recommendations
    """
    
    kg = EcologicalKnowledgeGraph()
    
    # Analyze the restoration site
    site_analysis = kg.analyze_restoration_site(latitude, longitude, environmental_conditions)
    
    return {
        'biogeographic_zone': site_analysis['biogeographic_zone'],
        'zone_characteristics': site_analysis['zone_characteristics'],
        'suitable_forest_types': site_analysis['suitable_forest_types'],
        'restoration_potential': site_analysis['restoration_assessment'],
        'restoration_recommendations': site_analysis['recommendations'],
        'methodology': 'Ecological Knowledge Graph + Biogeographic Analysis'
    }

# Quick coordinate lookup function
def quick_zone_lookup(latitude: float, longitude: float) -> str:
    """Quick lookup for biogeographic zone (lightweight version)"""
    kg = EcologicalKnowledgeGraph()
    return kg.map_coordinates_to_zone(latitude, longitude)

# Test function
if __name__ == "__main__":
    # Test the knowledge graph
    kg = EcologicalKnowledgeGraph()
    
    # Test coordinates (Bangalore)
    lat, lon = 12.9716, 77.5946
    zone = kg.map_coordinates_to_zone(lat, lon)
    print(f"🌍 Coordinates ({lat}, {lon}) mapped to: {zone}")
    
    # Test environmental analysis
    env_conditions = {
        'rainfall': 900,
        'temperature': 28,
        'elevation': 920
    }
    
    suitable_forests = kg.get_suitable_forest_types(zone, env_conditions)
    print(f"🌳 Suitable forest types:")
    for forest in suitable_forests[:3]:
        print(f"  - {forest['forest_type']}: {forest['suitability_score']:.2f} suitability")
    
    # Test full site analysis
    analysis = kg.analyze_restoration_site(lat, lon, env_conditions)
    print(f"🎯 Restoration potential: {analysis['restoration_assessment']['potential_grade']}")
    print(f"📈 Success probability: {analysis['restoration_assessment']['success_probability']}%")