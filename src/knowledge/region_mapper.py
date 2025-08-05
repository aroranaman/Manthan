"""
Region Mapper for Indian Ecological Zones
Maps coordinates to biogeographic zones, forest types, and administrative regions
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon
import pyproj
from functools import partial

class IndianRegionMapper:
    """
    Maps geographical coordinates to Indian ecological regions and administrative boundaries
    """
    
    def __init__(self):
        self.initialize_regions()
        self.initialize_state_boundaries()
    
    def initialize_regions(self):
        """
        Initialize biogeographic zones of India
        Based on Rodgers & Panwar (1988) classification
        """
        self.biogeographic_zones = {
            
            "Trans-Himalaya": {
                "states": ["Ladakh", "Himachal Pradesh (part)"],
                "coordinates": [(74.0, 32.0), (79.0, 32.0), (79.0, 36.0), (74.0, 36.0)],
                "elevation_range": (3000, 6000),
                "rainfall_range": (50, 200),
                "forest_types": ["Cold Desert"]
            },
            "Himalaya": {
                "states": ["Jammu & Kashmir", "Himachal Pradesh", "Uttarakhand", "Sikkim", "Arunachal Pradesh", "West Bengal (part)"],
                "coordinates": [(73.5, 27.0), (97.5, 27.0), (97.5, 35.0), (73.5, 35.0)],
                "elevation_range": (300, 8000),
                "rainfall_range": (600, 4000),
                "forest_types": ["Tropical Wet Evergreen", "Subtropical Pine", "Montane Wet Temperate", "Subalpine", "Alpine Meadow"]
            },
            "Desert": {
                "states": ["Rajasthan", "Gujarat (part)", "Punjab (part)", "Haryana (part)"],
                "coordinates": [(68.5, 23.0), (76.0, 23.0), (76.0, 30.0), (68.5, 30.0)],
                "elevation_range": (0, 500),
                "rainfall_range": (100, 600),
                "forest_types": ["Tropical Thorn", "Ravine Thorn", "Desert Dune Scrub"]
            },
            "Semi-Arid": {
                "states": ["Punjab", "Haryana", "Rajasthan (part)", "Gujarat (part)", 
                          "Madhya Pradesh (part)", "Uttar Pradesh (part)"],
                "coordinates": [(69.0, 21.0), (81.0, 21.0), (81.0, 31.0), (69.0, 31.0)],
                "elevation_range": (150, 900),
                "rainfall_range": (400, 800),
                "forest_types": ["Tropical Dry Deciduous", "Northern Dry Mixed Deciduous",
                               "Dry Grassland", "Ravine Scrub"]
            },
            "Western Ghats": {
                "states": ["Gujarat (part)", "Maharashtra", "Goa", "Karnataka", 
                          "Kerala", "Tamil Nadu (part)"],
                "coordinates": [(72.5, 8.0), (77.5, 8.0), (77.5, 21.0), (72.5, 21.0)],
                "elevation_range": (0, 2700),
                "rainfall_range": (1000, 7000),
                "forest_types": ["Tropical Wet Evergreen", "Tropical Semi-Evergreen",
                               "Tropical Moist Deciduous", "Montane Subtropical",
                               "Montane Temperate Shola"]
            },
            "Deccan Peninsula": {
                "states": ["Maharashtra (part)", "Karnataka (part)", "Andhra Pradesh",
                          "Telangana", "Tamil Nadu (part)"],
                "coordinates": [(74.0, 13.0), (81.0, 13.0), (81.0, 21.0), (74.0, 21.0)],
                "elevation_range": (300, 1200),
                "rainfall_range": (600, 1500),
                "forest_types": ["Tropical Dry Deciduous", "Southern Dry Mixed Deciduous",
                               "Dry Evergreen", "Dry Savannah"]
            },
            "Gangetic Plain": {
                "states": ["Uttar Pradesh", "Bihar", "West Bengal (part)", "Punjab (part)",
                          "Haryana (part)", "Delhi"],
                "coordinates": [(77.0, 23.5), (88.5, 23.5), (88.5, 31.0), (77.0, 31.0)],
                "elevation_range": (50, 300),
                "rainfall_range": (650, 2000),
                "forest_types": ["Tropical Moist Deciduous", "Dry Deciduous", 
                               "Swamp Forest", "Riparian Forest"]
            },
            "North-East": {
                "states": ["Assam", "Meghalaya", "Tripura", "Mizoram", "Manipur",
                          "Nagaland", "Arunachal Pradesh (part)"],
                "coordinates": [(89.5, 21.5), (97.5, 21.5), (97.5, 29.5), (89.5, 29.5)],
                "elevation_range": (50, 3000),
                "rainfall_range": (1000, 11000),
                "forest_types": ["Tropical Wet Evergreen", "Tropical Semi-Evergreen",
                               "Subtropical Broadleaf Hill", "Subtropical Pine",
                               "Montane Wet Temperate"]
            },
            "Islands": {
                "states": ["Andaman & Nicobar", "Lakshadweep"],
                "coordinates": [(92.0, 6.0), (94.0, 6.0), (94.0, 14.0), (92.0, 14.0)],
                "elevation_range": (0, 750),
                "rainfall_range": (1500, 3500),
                "forest_types": ["Tropical Wet Evergreen", "Littoral Forest", 
                               "Mangrove Forest"]
            },
            "Coasts": {
                "states": ["All coastal states"],
                "coordinates": "Coastal buffer zones",
                "elevation_range": (0, 100),
                "rainfall_range": (500, 3000),
                "forest_types": ["Mangrove", "Littoral", "Swamp", "Beach Forest"]
            }
        }
    
    def initialize_state_boundaries(self):
        """
        Initialize simplified state boundaries for quick lookup
        Using bounding boxes for performance, can be enhanced with actual polygons
        """
        self.state_bounds = {
            "Andhra Pradesh": {"lat": (12.5, 19.5), "lon": (76.5, 84.5)},
            "Arunachal Pradesh": {"lat": (26.5, 29.5), "lon": (91.5, 97.5)},
            "Assam": {"lat": (24.0, 28.0), "lon": (89.5, 96.0)},
            "Bihar": {"lat": (24.0, 27.5), "lon": (83.0, 88.5)},
            "Chhattisgarh": {"lat": (17.5, 24.5), "lon": (80.0, 84.5)},
            "Goa": {"lat": (14.8, 15.8), "lon": (73.6, 74.3)},
            "Gujarat": {"lat": (20.0, 24.5), "lon": (68.0, 74.5)},
            "Haryana": {"lat": (27.5, 31.0), "lon": (74.5, 77.5)},
            "Himachal Pradesh": {"lat": (30.5, 33.5), "lon": (75.5, 79.0)},
            "Jharkhand": {"lat": (21.5, 25.5), "lon": (83.0, 88.0)},
            "Karnataka": {"lat": (11.5, 18.5), "lon": (74.0, 78.5)},
            "Kerala": {"lat": (8.0, 13.0), "lon": (74.5, 77.5)},
            "Madhya Pradesh": {"lat": (21.0, 26.5), "lon": (74.0, 82.5)},
            "Maharashtra": {"lat": (15.5, 22.5), "lon": (72.5, 80.5)},
            "Manipur": {"lat": (23.5, 25.5), "lon": (93.0, 94.8)},
            "Meghalaya": {"lat": (25.0, 26.2), "lon": (89.5, 92.8)},
            "Mizoram": {"lat": (21.5, 24.5), "lon": (92.0, 93.8)},
            "Nagaland": {"lat": (25.0, 27.0), "lon": (93.0, 95.3)},
            "Odisha": {"lat": (17.5, 22.5), "lon": (81.5, 87.5)},
            "Punjab": {"lat": (29.5, 32.5), "lon": (73.5, 77.0)},
            "Rajasthan": {"lat": (23.0, 30.5), "lon": (69.5, 78.5)},
            "Sikkim": {"lat": (27.0, 28.0), "lon": (88.0, 89.0)},
            "Tamil Nadu": {"lat": (8.0, 13.5), "lon": (76.5, 80.5)},
            "Telangana": {"lat": (15.5, 19.5), "lon": (77.0, 81.5)},
            "Tripura": {"lat": (22.5, 24.5), "lon": (91.0, 92.5)},
            "Uttar Pradesh": {"lat": (23.5, 31.0), "lon": (77.0, 84.5)},
            "Uttarakhand": {"lat": (28.5, 31.5), "lon": (77.5, 81.5)},
            "West Bengal": {"lat": (21.5, 27.5), "lon": (85.5, 89.5)},
            "Delhi": {"lat": (28.4, 28.9), "lon": (76.8, 77.4)},
            "Ladakh": {"lat": (32.0, 37.0), "lon": (74.0, 80.0)},
            "Jammu & Kashmir": {"lat": (32.0, 35.0), "lon": (73.0, 77.0)}
        }
    
    def get_state_from_coordinates(self, lat: float, lon: float) -> Optional[str]:
        """
        Get state name from coordinates using bounding box check
        """
        for state, bounds in self.state_bounds.items():
            if (bounds["lat"][0] <= lat <= bounds["lat"][1] and 
                bounds["lon"][0] <= lon <= bounds["lon"][1]):
                return state
        return None
    
    def get_biogeographic_zone(self, lat: float, lon: float) -> Optional[str]:
        """
        Get biogeographic zone from coordinates
        """
        point = Point(lon, lat)
        
        # Check special cases first
        if lat > 32.0 and lon < 79.0:  # Trans-Himalaya
            return "Trans-Himalaya"
        
        if lat > 27.0 and lat < 35.0 and lon > 73.5:  # Himalaya
            return "Himalaya"
        
        # Western Ghats check (along western coast)
        if (lon > 72.5 and lon < 77.5 and lat > 8.0 and lat < 21.0 and
            self._is_western_ghats_region(lat, lon)):
            return "Western Ghats"
        
        # Desert region
        if lon < 76.0 and lat > 23.0 and lat < 30.0 and lon > 68.5:
            state = self.get_state_from_coordinates(lat, lon)
            if state in ["Rajasthan", "Gujarat"]:
                return "Desert"
        
        # North-East
        if lon > 89.5 and lat > 21.5 and lat < 29.5:
            return "North-East"
        
        # Default checks based on state and rainfall
        state = self.get_state_from_coordinates(lat, lon)
        if state:
            return self._infer_zone_from_state(state, lat, lon)
        
        return "Unknown"
    
    def _is_western_ghats_region(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates fall in Western Ghats region
        Using simplified logic based on distance from coast and elevation
        """
        # Simplified check - within 150km of western coast
        # In production, use actual Western Ghats boundary shapefile
        coastal_distance = abs(lon - 75.0)  # Approximate
        return coastal_distance < 2.5  # ~150-200km
    
    def _infer_zone_from_state(self, state: str, lat: float, lon: float) -> str:
        """
        Infer biogeographic zone from state and location
        """
        zone_mappings = {
            "Kerala": "Western Ghats",
            "Goa": "Western Ghats", 
            "Sikkim": "Himalaya",
            "Arunachal Pradesh": "North-East",
            "Assam": "North-East",
            "Meghalaya": "North-East",
            "Manipur": "North-East",
            "Mizoram": "North-East",
            "Nagaland": "North-East",
            "Tripura": "North-East",
            "Rajasthan": "Desert" if lon < 75.0 else "Semi-Arid",
            "Gujarat": "Desert" if lat > 23.0 and lon < 71.0 else "Semi-Arid",
            "Uttar Pradesh": "Gangetic Plain",
            "Bihar": "Gangetic Plain",
            "West Bengal": "Gangetic Plain" if lat > 24.0 else "Coasts",
            "Andhra Pradesh": "Deccan Peninsula" if lon < 80.0 else "Coasts",
            "Tamil Nadu": "Deccan Peninsula" if lon < 79.0 else "Coasts",
            "Karnataka": "Western Ghats" if lon < 76.0 else "Deccan Peninsula",
            "Maharashtra": "Western Ghats" if lon < 75.0 else "Deccan Peninsula",
            "Madhya Pradesh": "Semi-Arid" if lat > 24.0 else "Deccan Peninsula",
            "Chhattisgarh": "Deccan Peninsula",
            "Jharkhand": "Deccan Peninsula",
            "Odisha": "Deccan Peninsula" if lon < 85.0 else "Coasts",
            "Telangana": "Deccan Peninsula"
        }
        
        return zone_mappings.get(state, "Semi-Arid")
    
    def get_forest_types_for_location(self, lat: float, lon: float, 
                                    elevation: float = None,
                                    rainfall: float = None) -> List[str]:
        """
        Get possible forest types for a location based on zone and conditions
        """
        zone = self.get_biogeographic_zone(lat, lon)
        if zone in self.biogeographic_zones:
            zone_data = self.biogeographic_zones[zone]
            forest_types = zone_data["forest_types"]
            
            # Filter by elevation if provided
            if elevation is not None:
                elev_range = zone_data["elevation_range"]
                if elevation < elev_range[0] or elevation > elev_range[1]:
                    # Adjust forest types based on elevation
                    forest_types = self._adjust_for_elevation(forest_types, elevation)
            
            # Filter by rainfall if provided
            if rainfall is not None:
                rain_range = zone_data["rainfall_range"]
                if rainfall < rain_range[0]:
                    # More xeric types
                    forest_types = [ft for ft in forest_types if "Dry" in ft or "Thorn" in ft]
                elif rainfall > rain_range[1]:
                    # More mesic types
                    forest_types = [ft for ft in forest_types if "Wet" in ft or "Moist" in ft]
            
            return forest_types
        
        return ["Unknown Forest Type"]
    
    def _adjust_for_elevation(self, forest_types: List[str], elevation: float) -> List[str]:
        """
        Adjust forest types based on elevation
        """
        if elevation > 3000:
            return [ft for ft in forest_types if any(term in ft for term in 
                   ["Alpine", "Subalpine", "Temperate"])]
        elif elevation > 1500:
            return [ft for ft in forest_types if any(term in ft for term in 
                   ["Montane", "Subtropical", "Temperate", "Hill"])]
        elif elevation < 200:
            return [ft for ft in forest_types if any(term in ft for term in 
                   ["Littoral", "Mangrove", "Swamp", "Riparian"])]
        return forest_types
    
    def get_region_metadata(self, lat: float, lon: float) -> Dict:
        """
        Get comprehensive metadata for a location
        """
        state = self.get_state_from_coordinates(lat, lon)
        zone = self.get_biogeographic_zone(lat, lon)
        
        metadata = {
            "coordinates": {"lat": lat, "lon": lon},
            "state": state,
            "biogeographic_zone": zone,
            "forest_types": self.get_forest_types_for_location(lat, lon),
        }
        
        if zone in self.biogeographic_zones:
            zone_data = self.biogeographic_zones[zone]
            metadata.update({
                "elevation_range": zone_data["elevation_range"],
                "rainfall_range": zone_data["rainfall_range"],
                "zone_states": zone_data["states"]
            })
        
        return metadata
    
    def get_nearby_reference_sites(self, lat: float, lon: float, 
                                 radius_km: float = 50) -> List[Dict]:
        """
        Get nearby reference forest sites for comparison
        Can be enhanced with actual database of protected areas
        """
        # Placeholder for reference sites
        # In production, query from database of national parks, wildlife sanctuaries
        reference_sites = [
            {"name": "Local Forest Reserve", "distance_km": 15, "forest_type": "Tropical Dry Deciduous"},
            {"name": "Regional National Park", "distance_km": 35, "forest_type": "Tropical Moist Deciduous"}
        ]
        
        return reference_sites
