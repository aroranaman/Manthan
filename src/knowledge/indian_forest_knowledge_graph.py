"""
Indian Forest Knowledge Graph
Core knowledge graph implementation for ecological relationships
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional
from .indian_species_database import get_species_database, Species

class IndianForestKnowledgeGraph:
    """
    Knowledge graph representing Indian forest ecosystems
    Nodes: bioregions, forest types, layers, species
    Edges: ecological relationships
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.species_db = get_species_database()
        self._build_graph()
    
    def _build_graph(self):
        """Build the complete knowledge graph"""
        self._add_bioregions()
        self._add_forest_types()
        self._add_canopy_layers()
        self._add_species()
        self._create_relationships()
    
    def _add_bioregions(self):
        """Add biogeographic regions of India"""
        bioregions = [
            {"id": "western_ghats", "name": "Western Ghats", "type": "bioregion"},
            {"id": "eastern_ghats", "name": "Eastern Ghats", "type": "bioregion"},
            {"id": "himalayas", "name": "Himalayas", "type": "bioregion"},
            {"id": "indo_gangetic", "name": "Indo-Gangetic Plain", "type": "bioregion"},
            {"id": "thar_desert", "name": "Thar Desert", "type": "bioregion"},
            {"id": "deccan_plateau", "name": "Deccan Plateau", "type": "bioregion"},
            {"id": "northeast", "name": "Northeast India", "type": "bioregion"},
            {"id": "coastal", "name": "Coastal Regions", "type": "bioregion"},
            {"id": "andaman", "name": "Andaman & Nicobar", "type": "bioregion"}
        ]
        
        for region in bioregions:
            self.graph.add_node(region["id"], **region)
    
    def _add_forest_types(self):
        """Add forest types and connect to bioregions"""
        forest_mappings = {
            "western_ghats": [
                {"id": "trop_wet_evergreen", "name": "Tropical Wet Evergreen", 
                 "rainfall": (2000, 7000), "temperature": (18, 32)},
                {"id": "trop_semi_evergreen", "name": "Tropical Semi-Evergreen",
                 "rainfall": (1500, 3000), "temperature": (20, 35)},
                {"id": "shola", "name": "Montane Shola",
                 "rainfall": (1500, 5000), "temperature": (10, 25)}
            ],
            "deccan_plateau": [
                {"id": "trop_dry_deciduous", "name": "Tropical Dry Deciduous",
                 "rainfall": (600, 1500), "temperature": (20, 45)},
                {"id": "trop_moist_deciduous", "name": "Tropical Moist Deciduous",
                 "rainfall": (1000, 2000), "temperature": (20, 40)}
            ],
            "thar_desert": [
                {"id": "desert_thorn", "name": "Desert Thorn Forest",
                 "rainfall": (100, 600), "temperature": (5, 50)},
                {"id": "tropical_thorn", "name": "Tropical Thorn",
                 "rainfall": (200, 700), "temperature": (10, 48)}
            ],
            "himalayas": [
                {"id": "montane_temperate", "name": "Montane Wet Temperate",
                 "rainfall": (1500, 3500), "temperature": (0, 25)},
                {"id": "sub_alpine", "name": "Sub-Alpine",
                 "rainfall": (800, 2000), "temperature": (-5, 20)},
                {"id": "subtropical_pine", "name": "Subtropical Pine",
                 "rainfall": (1000, 2500), "temperature": (5, 30)}
            ],
            "coastal": [
                {"id": "mangrove", "name": "Mangrove Forest",
                 "rainfall": (1000, 3000), "temperature": (20, 35)},
                {"id": "littoral", "name": "Littoral Forest",
                 "rainfall": (800, 3000), "temperature": (20, 38)}
            ],
            "northeast": [
                {"id": "ne_trop_evergreen", "name": "Tropical Wet Evergreen",
                 "rainfall": (2000, 11000), "temperature": (15, 35)},
                {"id": "ne_subtropical", "name": "Subtropical Broadleaf Hill",
                 "rainfall": (1500, 4000), "temperature": (10, 30)}
            ]
        }
        
        for bioregion, forest_types in forest_mappings.items():
            for ft in forest_types:
                self.graph.add_node(ft["id"], type="forest_type", **ft)
                self.graph.add_edge(bioregion, ft["id"], relationship="contains")
    
    def _add_canopy_layers(self):
        """Add forest canopy layers"""
        layers = [
            {"id": "emergent", "name": "Emergent Layer", "height": "35-45m"},
            {"id": "canopy", "name": "Canopy Layer", "height": "20-35m"},
            {"id": "understory", "name": "Understory Layer", "height": "5-20m"},
            {"id": "ground", "name": "Ground Layer", "height": "0-5m"}
        ]
        
        for layer in layers:
            self.graph.add_node(layer["id"], type="layer", **layer)
        
        # Connect layers to all forest types
        forest_types = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "forest_type"]
        for ft in forest_types:
            for layer in layers:
                self.graph.add_edge(ft, layer["id"], relationship="has_layer")
    
    def _add_species(self):
        """Add species nodes and properties"""
        for sci_name, species in self.species_db.items():
            # Create a clean dictionary of attributes
            node_attrs = {
                'type': 'species',
                'scientific_name': sci_name,
                'common_name': species.common_name,
                'family': species.family,
                'layer': species.canopy_layer,
                'carbon_seq': species.carbon_sequestration,
                'native_regions': species.native_regions,
                'forest_types': species.forest_types,
                'growth_rate': species.growth_rate,
                'rainfall_min': species.rainfall_min,
                'rainfall_max': species.rainfall_max,
                'temperature_min': species.temperature_min,
                'temperature_max': species.temperature_max,
                'soil_ph_min': species.soil_ph_min,
                'soil_ph_max': species.soil_ph_max,
                'elevation_min': species.elevation_min,
                'elevation_max': species.elevation_max,
                'ecological_value': species.ecological_value,
                'planting_notes': species.planting_notes
            }
            
            # Add the node with clean attributes
            self.graph.add_node(sci_name, **node_attrs)

    
    def _create_relationships(self):
        """Create ecological relationships between nodes"""
        # Connect species to their native bioregions
        bioregion_mapping = {
            "Western Ghats": "western_ghats",
            "Eastern Ghats": "eastern_ghats",
            "Himalayas": "himalayas",
            "Central India": "deccan_plateau",
            "Northern India": "indo_gangetic",
            "Rajasthan": "thar_desert",
            "Northeast India": "northeast",
            "Coastal India": "coastal",
            "Andaman": "andaman"
        }
        
        for sci_name, species in self.species_db.items():
            # Connect to bioregions
            for region in species.native_regions:
                if region in bioregion_mapping:
                    bioregion_id = bioregion_mapping[region]
                    if self.graph.has_node(bioregion_id):
                        self.graph.add_edge(bioregion_id, sci_name, relationship="native_species")
            
            # Connect to forest types
            forest_type_mapping = {
                "Tropical Wet Evergreen": ["trop_wet_evergreen", "ne_trop_evergreen"],
                "Tropical Semi-Evergreen": "trop_semi_evergreen",
                "Tropical Dry Deciduous": "trop_dry_deciduous",
                "Tropical Moist Deciduous": "trop_moist_deciduous",
                "Desert": "desert_thorn",
                "Tropical Thorn": "tropical_thorn",
                "Montane Wet Temperate": "montane_temperate",
                "Mangrove": "mangrove"
            }
            
            for ft in species.forest_types:
                if ft in forest_type_mapping:
                    ft_ids = forest_type_mapping[ft]
                    if isinstance(ft_ids, str):
                        ft_ids = [ft_ids]
                    for ft_id in ft_ids:
                        if self.graph.has_node(ft_id):
                            self.graph.add_edge(ft_id, sci_name, relationship="contains_species")
            
            # Connect to canopy layers
            if species.canopy_layer and self.graph.has_node(species.canopy_layer):
                self.graph.add_edge(species.canopy_layer, sci_name, relationship="occupies_layer")
    
    def get_ecological_structure(self, forest_type: str, bioregion: str = None) -> Dict:
        """Get the ecological structure for a forest type"""
        structures = {
            "Tropical Wet Evergreen": {
                "layer_proportions": {"emergent": 15, "canopy": 55, "understory": 25, "ground": 5},
                "description": "Dense, multi-layered forest with high biodiversity"
            },
            "Tropical Dry Deciduous": {
                "layer_proportions": {"emergent": 10, "canopy": 50, "understory": 35, "ground": 5},
                "description": "Seasonally deciduous with open canopy in dry season"
            },
            "Desert Thorn Forest": {
                "layer_proportions": {"emergent": 5, "canopy": 35, "understory": 45, "ground": 15},
                "description": "Sparse canopy with thorny species, more shrub layer"
            },
            "Montane Wet Temperate": {
                "layer_proportions": {"emergent": 20, "canopy": 50, "understory": 25, "ground": 5},
                "description": "Temperate forest with conifers and broadleaf mix"
            },
            "Mangrove Forest": {
                "layer_proportions": {"emergent": 0, "canopy": 70, "understory": 25, "ground": 5},
                "description": "Dense canopy adapted to saline conditions"
            }
        }
        
        # Default structure
        default = {
            "layer_proportions": {"emergent": 15, "canopy": 50, "understory": 30, "ground": 5},
            "description": "Mixed forest structure"
        }
        
        return structures.get(forest_type, default)
    
    def get_species_for_forest_type(
        self,
        forest_type: str,
        state: str = None,
        rainfall: float = None,
        soil_ph: float = None
    ) -> Dict[str, List[Dict]]:
        """Get species recommendations organized by canopy layer"""
        
        # Map forest type names to graph IDs
        ft_mapping = {
            "Tropical Wet Evergreen": ["trop_wet_evergreen", "ne_trop_evergreen"],
            "Tropical Dry Deciduous": "trop_dry_deciduous",
            "Tropical Moist Deciduous": "trop_moist_deciduous",
            "Desert Thorn Forest": "desert_thorn",
            "Tropical Thorn": "tropical_thorn",
            "Montane Wet Temperate": "montane_temperate",
            "Mangrove Forest": "mangrove"
        }
        
        ft_ids = ft_mapping.get(forest_type, [])
        if isinstance(ft_ids, str):
            ft_ids = [ft_ids]
        
        species_by_layer = {
            "emergent": [],
            "canopy": [],
            "understory": [],
            "ground": []
        }
        
        # Get species connected to this forest type
        for ft_id in ft_ids:
            if not self.graph.has_node(ft_id):
                continue
                
            # Get all species for this forest type
            species_nodes = [
                n for n in self.graph.successors(ft_id)
                if self.graph.nodes[n].get("type") == "species"
            ]
            
            for sp_node in species_nodes:
                sp_data = self.graph.nodes[sp_node]
                species = self.species_db.get(sp_node)
                
                if not species:
                    continue
                
                # Apply filters
                if rainfall and not (species.rainfall_min <= rainfall <= species.rainfall_max):
                    continue
                    
                if soil_ph and not (species.soil_ph_min <= soil_ph <= species.soil_ph_max):
                    continue
                
                # Create species info dict
                sp_info = {
                    "scientific_name": species.scientific_name,
                    "common_name": species.common_name,
                    "family": species.family,
                    "carbon_sequestration": species.carbon_sequestration,
                    "ecological_value": species.ecological_value,
                    "planting_notes": species.planting_notes,
                    "growth_rate": species.growth_rate
                }
                
                # Add to appropriate layer
                layer = species.canopy_layer
                if layer in species_by_layer:
                    species_by_layer[layer].append(sp_info)
        
        return species_by_layer
    
    def get_bioregion_info(self, bioregion_id: str) -> Dict:
        """Get information about a bioregion"""
        if not self.graph.has_node(bioregion_id):
            return {}
        
        info = dict(self.graph.nodes[bioregion_id])
        
        # Get forest types in this bioregion
        forest_types = [
            n for n in self.graph.successors(bioregion_id)
            if self.graph.nodes[n].get("type") == "forest_type"
        ]
        
        info["forest_types"] = [
            self.graph.nodes[ft]["name"] for ft in forest_types
        ]
        
        # Get native species count
        native_species = [
            n for n in self.graph.successors(bioregion_id)
            if self.graph.nodes[n].get("type") == "species"
        ]
        
        info["native_species_count"] = len(native_species)
        
        return info
