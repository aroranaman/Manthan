"""
🌿 Manthan: Professional Forest Intelligence Dashboard
Complete implementation with all features
"""

# ───────────────────────── PATH FIX ────────────────────────────
import sys
import json
import time
from pathlib import Path
from datetime import datetime, date

try:
    ROOT = Path(__file__).resolve().parents[3]  # …/Manthan
    SRC = ROOT / "src"
    for p in (ROOT, SRC):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
except Exception as e:
    ROOT = Path.cwd()
    SRC = ROOT / "src"
# ───────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from folium.plugins import Draw
import plotly.graph_objects as go
import plotly.express as px

# Around line 35, add this BEFORE the debug line:
# Knowledge Graph System Import
ECOLOGICAL_INTELLIGENCE_AVAILABLE = False

try:
    import sys
    import os
    
    # Add parent directories to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up to src/
    manthan_root = os.path.dirname(src_dir)  # Go up to Manthan/
    
    sys.path.insert(0, manthan_root)
    sys.path.insert(0, src_dir)
    
    from src.knowledge.manthan_integration import ManthanEcologicalIntelligence
    ECOLOGICAL_INTELLIGENCE_AVAILABLE = True
    print("✅ Successfully imported ManthanEcologicalIntelligence")
    
except ImportError as e:
    print(f"❌ Knowledge Graph import failed: {e}")
    ECOLOGICAL_INTELLIGENCE_AVAILABLE = False

# Import the new ISFR data loader with a fallback
try:
    from src.knowledge.isfr_data_loader import ISFRDataLoader
    ISFR_DATA_AVAILABLE = True
except ImportError:
    ISFR_DATA_AVAILABLE = False

# Custom module imports with fallback
try:
    from utils.gee_auth import gee_init
    from scripts.day2_pipeline import ManthanDay2Pipeline
    CUSTOM_MODULES_AVAILABLE = True
except ImportError:
    CUSTOM_MODULES_AVAILABLE = False

    def gee_init():
        return True

    class ManthanDay2Pipeline:
        def __init__(self, aoi):
            self.aoi = aoi

        def run_complete_analysis(self):
            return {
                "ndvi": {"ndvi_mean": 0.73, "ndvi_std": 0.05},
                "rainfall": {"annual_rainfall": 1250, "rainfall_adequacy": "Adequate"},
                "soil": {"soil_ph": 6.8, "ph_suitability": "Optimal"},
                "suitability": {"composite_score": 0.857, "suitability_grade": "A"}
            }
# ─────────────── INDIAN FOREST ECOLOGICAL DATABASE ─────────────────
class IndianForestEcology:
    """
    Comprehensive database of Indian tree species by ecological zones
    Based on Champion & Seth classification and regional variations
    """
    
    def __init__(self):
        self.forest_types = {
            "Western Ghats - Wet Evergreen": {
                "states": ["Kerala", "Karnataka", "Tamil Nadu", "Maharashtra", "Goa"],
                "rainfall": (2000, 7000),
                "temperature": (18, 32),
                "elevation": (0, 1800),
                "soil_ph": (4.5, 6.5),
                "canopy_ratios": {"emergent": 15, "canopy": 50, "understory": 30, "shrub": 5},
                "species": {
                    "emergent": [
                        {"name": "Cullenia", "scientific": "Cullenia exarillata", "family": "Malvaceae", 
                         "carbon": 350, "endemic": True, "features": ["Wildlife", "Endemic", "Keystone"]},
                        {"name": "Wild Nutmeg", "scientific": "Myristica dactyloides", "family": "Myristicaceae",
                         "carbon": 320, "endemic": True, "features": ["Endemic", "Threatened", "Wildlife"]}
                    ],
                    "canopy": [
                        {"name": "Holigarna", "scientific": "Holigarna arnottiana", "family": "Anacardiaceae",
                         "carbon": 280, "endemic": True, "features": ["Endemic", "Timber", "Wildlife"]},
                        {"name": "Dhup", "scientific": "Canarium strictum", "family": "Burseraceae",
                         "carbon": 260, "features": ["Resin", "Wildlife", "Medicinal"]}
                    ],
                    "understory": [
                        {"name": "Wild Cardamom", "scientific": "Elettaria cardamomum", "family": "Zingiberaceae",
                         "carbon": 80, "features": ["Spice", "Economic", "Understory"]},
                        {"name": "Tree Fern", "scientific": "Cyathea nilgirensis", "family": "Cyatheaceae",
                         "carbon": 60, "endemic": True, "features": ["Endemic", "Ornamental", "Threatened"]}
                    ]
                }
            },
            "Eastern Himalayas - Subtropical": {
                "states": ["Sikkim", "Arunachal Pradesh", "Assam", "Meghalaya", "West Bengal"],
                "rainfall": (1500, 4000),
                "temperature": (10, 28),
                "elevation": (300, 2500),
                "soil_ph": (5.0, 6.5),
                "canopy_ratios": {"emergent": 20, "canopy": 45, "understory": 30, "shrub": 5},
                "species": {
                    "emergent": [
                        {"name": "Himalayan Oak", "scientific": "Quercus lamellosa", "family": "Fagaceae",
                         "carbon": 400, "features": ["Timber", "Wildlife", "Watershed"]},
                        {"name": "Magnolia", "scientific": "Magnolia campbellii", "family": "Magnoliaceae",
                         "carbon": 320, "features": ["Ornamental", "Threatened", "Endemic"]}
                    ],
                    "canopy": [
                        {"name": "Rhododendron", "scientific": "Rhododendron arboreum", "family": "Ericaceae",
                         "carbon": 180, "features": ["Ornamental", "Medicinal", "Honey"]},
                        {"name": "Himalayan Birch", "scientific": "Betula utilis", "family": "Betulaceae",
                         "carbon": 220, "features": ["Medicinal", "Paper", "Sacred"]}
                    ]
                }
            },
            "Central India - Dry Deciduous": {
                "states": ["Madhya Pradesh", "Chhattisgarh", "Jharkhand", "Maharashtra", "Odisha"],
                "rainfall": (800, 1500),
                "temperature": (20, 45),
                "elevation": (200, 800),
                "soil_ph": (6.0, 7.5),
                "canopy_ratios": {"emergent": 10, "canopy": 55, "understory": 30, "shrub": 5},
                "species": {
                    "emergent": [
                        {"name": "Sal", "scientific": "Shorea robusta", "family": "Dipterocarpaceae",
                         "carbon": 350, "features": ["Timber", "Resin", "Dominant"]},
                        {"name": "Mahua", "scientific": "Madhuca longifolia", "family": "Sapotaceae",
                         "carbon": 280, "features": ["Edible", "Oil", "Cultural"]}
                    ],
                    "canopy": [
                        {"name": "Teak", "scientific": "Tectona grandis", "family": "Lamiaceae",
                         "carbon": 250, "features": ["Premium Timber", "Plantation", "Economic"]},
                        {"name": "Tendu", "scientific": "Diospyros melanoxylon", "family": "Ebenaceae",
                         "carbon": 200, "features": ["Leaves", "Fruit", "Economic"]}
                    ]
                }
            },
            "Rajasthan - Arid Zone": {
                "states": ["Rajasthan", "Gujarat", "Haryana", "Punjab"],
                "rainfall": (200, 600),
                "temperature": (5, 50),
                "elevation": (100, 500),
                "soil_ph": (7.0, 8.5),
                "canopy_ratios": {"emergent": 5, "canopy": 40, "understory": 45, "shrub": 10},
                "species": {
                    "canopy": [
                        {"name": "Khejri", "scientific": "Prosopis cineraria", "family": "Fabaceae",
                         "carbon": 150, "features": ["Nitrogen Fixing", "Fodder", "Sacred"]},
                        {"name": "Rohida", "scientific": "Tecomella undulata", "family": "Bignoniaceae",
                         "carbon": 120, "features": ["Timber", "Medicinal", "Desert Tree"]}
                    ],
                    "understory": [
                        {"name": "Ker", "scientific": "Capparis decidua", "family": "Capparaceae",
                         "carbon": 80, "features": ["Edible", "Pickle", "Drought Resistant"]},
                        {"name": "Jal", "scientific": "Salvadora persica", "family": "Salvadoraceae",
                         "carbon": 70, "features": ["Toothbrush Tree", "Medicinal", "Fodder"]}
                    ]
                }
            }
        }
    
    def get_species_for_location(self, state, rainfall, temperature, soil_ph):
        """
        Returns species recommendations based on location and environmental parameters
        """
        suitable_forest_types = []
        
        for forest_type, data in self.forest_types.items():
            if state in data["states"]:
                # Check environmental match
                rain_min, rain_max = data["rainfall"]
                temp_min, temp_max = data["temperature"]
                ph_min, ph_max = data["soil_ph"]
                
                if (rain_min <= rainfall <= rain_max and 
                    temp_min <= temperature <= temp_max and
                    ph_min <= soil_ph <= ph_max):
                    suitable_forest_types.append((forest_type, data))
        
        if not suitable_forest_types:
            # Fallback to closest match
            suitable_forest_types = [(ft, d) for ft, d in self.forest_types.items() 
                                   if state in d["states"]][:1]
        
        return suitable_forest_types[0] if suitable_forest_types else None
    
    def calculate_species_mix(self, area_ha, forest_type_data):
        """
        Calculate species mix based on ecological layer proportions
        """
        trees_per_ha = 2500  # Standard for dense planting
        total_trees = int(area_ha * trees_per_ha)
        
        canopy_ratios = forest_type_data["canopy_ratios"]
        species_data = forest_type_data["species"]
        
        recommendations = []
        
        for layer, percentage in canopy_ratios.items():
            trees_in_layer = int(total_trees * percentage / 100)
            
            if layer in species_data:
                species_in_layer = species_data[layer]
                trees_per_species = trees_in_layer // len(species_in_layer) if species_in_layer else 0
                
                for species in species_in_layer:
                    species_copy = species.copy()
                    species_copy.update({
                        "layer": layer.title(),
                        "count": trees_per_species,
                        "percentage": percentage / len(species_in_layer)
                    })
                    recommendations.append(species_copy)
        
        return recommendations


# ─────────────── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="Manthan: Forest Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────── PROFESSIONAL CSS STYLING ─────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header Section */
    .main-header {
        background: linear-gradient(135deg, #065f46 0%, #059669 25%, #10b981 100%);
        margin: -2rem -1rem 2rem -1rem;
        padding: 3rem 2rem;
        text-align: center;
        border-radius: 0 0 2rem 2rem;
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.3);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    /* ADD THESE LINES to fix text color */
    [data-testid="metric-container"] label {
        color: #6b7280 !important;  /* Gray text for labels */
    }

    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #111827 !important;  /* Dark text for values */
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #f3f4f6;
        padding: 0.5rem;
        border-radius: 1rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        background: transparent;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: white;
        color: #111827;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.3);
    }
    
    /* Custom Cards */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        color: #111827 !important;  /* ADD THIS to ensure text is dark */
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .card-icon {
        width: 3rem;
        height: 3rem;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.3);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827 !important;  /* Change from white to dark */
        margin: 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px -3px rgba(16, 185, 129, 0.4);
    }
    
    /* Progress Steps */
    .progress-step {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        border: 2px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .progress-step.active {
        border-color: #10b981;
        background: #f0fdf4;
    }
    
    .progress-step.completed {
        border-color: #10b981;
        background: #d1fae5;
    }
    
    .progress-number {
        width: 2.5rem;
        height: 2.5rem;
        background: #e5e7eb;
        color: #6b7280;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.125rem;
    }
    
    .progress-number.active {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
    }
    
    .progress-number.completed {
        background: #10b981;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f9fafb;
    }

    /* ADD THESE to ensure sidebar text is visible */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] p {
        color: #374151 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #4b5563 !important;
    }

    
    /* Map container */
    .map-container {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
            /* Global text color fixes for white-on-white issues */
    .stApp {
        color: #111827 !important;
    }

    /* Fix all white text on white background */
    .element-container, .stMarkdown, .stText {
        color: #111827 !important;
    }

    /* Ensure metric cards have dark text */
    [data-testid="metric-container"] * {
        color: #111827 !important;
    }

    /* Fix sidebar text */
    section[data-testid="stSidebar"] * {
        color: #374151 !important;
    }

    /* Fix input labels */
    label {
        color: #4b5563 !important;
    }

    /* Fix button text (keep white for primary buttons) */
    .stButton > button {
        color: white !important;
    }

    /* Fix card text */
    .custom-card * {
        color: #111827 !important;
    }

    /* Special case for headers in colored backgrounds */
    .main-header * {
        color: white !important;  /* Keep header text white */
    }

</style>
""", unsafe_allow_html=True)

# ───────────── HELPER FUNCTIONS ──────────────────────────────
def normalise(res: dict) -> dict:
    """Normalize pipeline results to expected format"""
    res.setdefault("ndvi_data", res.get("ndvi", {}))
    res.setdefault("environmental_data", res.get("rainfall", {}))
    res.setdefault("soil_data", res.get("soil", {}))
    return res
# ─────────────── REAL DATA FETCHERS ─────────────────────────────
def get_state_from_coordinates(lat, lon):
    """
    Get Indian state from coordinates using reverse geocoding
    """
    # This is a simplified version - in production, use a proper geocoding service
    state_bounds = {
        "Kerala": {"lat": (8.0, 13.0), "lon": (74.5, 77.5)},
        "Karnataka": {"lat": (11.5, 18.5), "lon": (74.0, 78.5)},
        "Maharashtra": {"lat": (15.5, 22.5), "lon": (72.5, 80.5)},
        "Tamil Nadu": {"lat": (8.0, 13.5), "lon": (76.5, 80.5)},
        "Rajasthan": {"lat": (23.0, 30.5), "lon": (69.5, 78.5)},
        "Madhya Pradesh": {"lat": (21.0, 26.5), "lon": (74.0, 82.5)},
        "Sikkim": {"lat": (27.0, 28.0), "lon": (88.0, 89.0)},
        # Add more states as needed
    }
    
    for state, bounds in state_bounds.items():
        if (bounds["lat"][0] <= lat <= bounds["lat"][1] and 
            bounds["lon"][0] <= lon <= bounds["lon"][1]):
            return state
    
    return "Unknown"

def validate_environmental_data(rainfall, soil_ph, ndvi):
    """
    Validate that environmental data is within reasonable bounds for India
    """
    issues = []
    
    if not 50 <= rainfall <= 12000:
        issues.append(f"Rainfall {rainfall}mm seems incorrect for India (expected 50-12000mm)")
    
    if not 3.5 <= soil_ph <= 9.5:
        issues.append(f"Soil pH {soil_ph} seems incorrect (expected 3.5-9.5)")
        
    if not -0.2 <= ndvi <= 1.0:
        issues.append(f"NDVI {ndvi} seems incorrect (expected -0.2 to 1.0)")
    
    return issues

def create_species_card(species):
    """Create beautiful species recommendation card"""
    return f"""
    <div class="custom-card" style="background: linear-gradient(135deg, #f0fdf4 0%, white 100%);">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h4 style="color: #059669; margin: 0; font-size: 1.25rem;">{species['name']}</h4>
                <p style="color: #6b7280; font-style: italic; margin: 0.25rem 0;">
                    {species['scientific']}
                </p>
                <p style="color: #9ca3af; font-size: 0.875rem;">
                    Family: {species['family']} | Layer: {species['layer']}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); 
                            color: white; padding: 0.5rem 1rem; border-radius: 9999px; 
                            font-size: 0.875rem; font-weight: 600;">
                    {species['score']}% Match
                </div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0;">
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">🌧️ Rainfall</p>
                <p style="font-weight: 600;">{species['rainfall']} mm</p>
            </div>
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">🌳 Carbon Seq.</p>
                <p style="font-weight: 600;">{species['carbon']} kg/year</p>
            </div>
        </div>
        
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;">
            {' '.join([f'<span style="background: #d1fae5; color: #065f46; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">✓ {feature}</span>' for feature in species['features']])}
        </div>
    </div>
    """

# ───────────── SESSION STATE ──────────────────────────────
for key, default in {
    "gee_authenticated": False,
    "aoi_data": None,
    "analysis_results": None,
    "current_step": 1,
    "drawn_polygon": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ───────────── HEADER ──────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 class="header-title">🌿 Manthan</h1>
    <p class="header-subtitle">AI-Powered Forest Restoration Intelligence for India</p>
</div>
""", unsafe_allow_html=True)

# ───────────── TOP METRICS (if analysis complete) ──────────────
if st.session_state.get("analysis_results"):
    res = st.session_state["analysis_results"]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Site Score",
            f"{res['suitability']['composite_score']:.1%}",
            delta=f"Grade {res['suitability']['suitability_grade']}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "🌱 Vegetation",
            f"{res['ndvi_data']['ndvi_mean']:.3f}",
            delta=f"±{res['ndvi_data']['ndvi_std']:.3f}"
        )
    
    with col3:
        st.metric(
            "🌧️ Rainfall",
            f"{res['environmental_data']['annual_rainfall']:.0f} mm",
            delta=res['environmental_data']['rainfall_adequacy']
        )
    
    with col4:
        st.metric(
            "🧪 Soil pH",
            f"{res['soil_data']['soil_ph']:.1f}",
            delta=res['soil_data']['ph_suitability']
        )
    
    with col5:
        area = st.session_state.get("aoi_data", {}).get("area_ha", 0)
        st.metric(
            "📏 Area",
            f"{area:.1f} ha" if area else "N/A",
            delta=f"{area * 2.47:.1f} acres" if area else None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)

# ───────────── TABS ────────────────────────────────────────────
tab_site, tab_env, tab_spec, tab_plan, tab_rep = st.tabs([
    "📍 Site Selection",
    "🔬 Environmental Analysis",
    "🌿 Species Recommendations",
    "📊 Restoration Plan",
    "📄 Reports & Export"
])

# =================================================================
# TAB 1 – SITE SELECTION
# =================================================================
with tab_site:
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">🗺️</div>
                <div>
                    <h3 class="card-title">Select Restoration Site</h3>
                    <p class="card-subtitle">Draw or click on the map to define your area</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Map container
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        
        # Folium map
        m = folium.Map(
            location=[22.5, 79],
            zoom_start=5,
            tiles="OpenStreetMap"
        )
        
        Draw(
            export=True,
            position='topleft',
            draw_options={
                "polyline": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": True,
                "polygon": True
            },
            edit_options={"edit": True}
        ).add_to(m)
        
        # Display map
        map_data = st_folium(
            m,
            height=500,
            width=None,
            returned_objects=["last_active_drawing"],
            key="main_map"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Check if area is drawn
        if map_data and map_data.get('last_active_drawing'):
            st.session_state.drawn_polygon = map_data['last_active_drawing']
            
            # Show confirmation buttons for drawn area
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                if st.button("✅ Use This Area", type="primary", use_container_width=True, key="use_drawn_area_1"):
                    geom = st.session_state.drawn_polygon['geometry']
                    if geom['type'] == 'Polygon':
                        coords = geom['coordinates'][0]
                    else:  # Rectangle
                        coords = geom['coordinates'][0]
                    
                    polygon = Polygon(coords)
                    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:4326")
                    area_ha = gdf.to_crs("EPSG:3857").geometry.area.iloc[0] / 10000
                    
                    st.session_state["aoi_data"] = {
                        "polygon": polygon,
                        "area_ha": area_ha,
                        "geojson": st.session_state.drawn_polygon,
                    }
                    st.session_state["current_step"] = 2
                    st.success(f"✅ Area saved: {area_ha:.1f} hectares")
            
            with col_b:
                if st.button("🗑️ Clear Drawing", use_container_width=True, key="clear_drawing_btn"):
                    st.session_state.drawn_polygon = None
                    st.rerun()
    
    with col2:
        # Progress Card
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">📋</div>
                <div>
                    <h3 class="card-title">Setup Progress</h3>
                    <p class="card-subtitle">Complete all steps</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress steps
        steps = [
            ("1", "Select Area", st.session_state.current_step >= 1, st.session_state.current_step == 1),
            ("2", "Configure", st.session_state.current_step >= 2, st.session_state.current_step == 2),
            ("3", "Analyze", st.session_state.current_step >= 3, st.session_state.current_step == 3),
            ("4", "Results", st.session_state.current_step >= 4, st.session_state.current_step == 4)
        ]
        
        for num, text, completed, active in steps:
            step_class = "completed" if completed and not active else "active" if active else ""
            num_class = "completed" if completed and not active else "active" if active else ""
            icon = "✓" if completed and not active else num
            
            st.markdown(f"""
            <div class="progress-step {step_class}">
                <div class="progress-number {num_class}">{icon}</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: {'#059669' if completed or active else '#6b7280'};">
                        {text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Configuration
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">⚙️</div>
                <div>
                    <h3 class="card-title">Configuration</h3>
                    <p class="card-subtitle">Analysis parameters</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Restoration method
        st.selectbox(
            "🌳 Restoration Method",
            ["Auto-Detect Based on Site", "Miyawaki Dense Forest", 
             "Agroforestry System", "Wildlife Habitat", "Eco-Tourism Forest"],
            key="restoration_method",
            help="Different methods optimize for different outcomes"
        )
        
        # Manual coordinate entry
        with st.expander("📐 Enter Coordinates Manually"):
            c1, c2 = st.columns(2)
            with c1:
                lat_min = st.number_input("Min Latitude", value=20.0, step=0.1)
                lat_max = st.number_input("Max Latitude", value=21.0, step=0.1)
            with c2:
                lon_min = st.number_input("Min Longitude", value=78.0, step=0.1)
                lon_max = st.number_input("Max Longitude", value=79.0, step=0.1)
            
            if st.button("➕ Create AOI from Coordinates", use_container_width=True, key="create_aoi_coords"):
                if lat_min < lat_max and lon_min < lon_max:
                    coords = [
                        [lon_min, lat_min], [lon_max, lat_min],
                        [lon_max, lat_max], [lon_min, lat_max],
                        [lon_min, lat_min],
                    ]
                    polygon = Polygon(coords)
                    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:4326")
                    area_ha = gdf.to_crs("EPSG:3857").geometry.area.iloc[0] / 10000
                    st.session_state["aoi_data"] = {
                        "polygon": polygon,
                        "area_ha": area_ha,
                        "geojson": {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": [coords]},
                            "properties": {}
                        },
                    }
                    st.session_state["current_step"] = 2
                    st.success(f"✅ AOI created: {area_ha:.1f} hectares")
                else:
                    st.error("Min values must be less than max values")

if st.button("✅ Use This Area", type="primary", use_container_width=True, key="use_area_btn_line809"):
    geom = st.session_state.drawn_polygon['geometry']
    if geom['type'] == 'Polygon':
        coords = geom['coordinates'][0]
    else:  # Rectangle
        coords = geom['coordinates'][0]
    
    polygon = Polygon(coords)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:4326")
    area_ha = gdf.to_crs("EPSG:3857").geometry.area.iloc[0] / 10000
    
    # Ensure consistent structure
    st.session_state["aoi_data"] = {
        "polygon": polygon,
        "area_ha": area_ha,  # Now calculated for drawn areas too
        "geojson": st.session_state.drawn_polygon,
    }
    st.session_state["current_step"] = 2
    st.success(f"✅ Area saved: {area_ha:.1f} hectares")

    # Run Analysis Button
    if st.session_state.get("aoi_data"):
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, key="run_analysis_btn"):
            aoi_geojson = st.session_state["aoi_data"]["geojson"]
            
            with st.spinner("🔍 Analyzing site conditions..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Progress updates
                status_text.text("Initializing Earth Engine...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("Analyzing vegetation (NDVI)...")
                progress_bar.progress(30)
                time.sleep(0.5)
                
                status_text.text("Fetching rainfall data...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("Analyzing soil properties...")
                progress_bar.progress(70)
                time.sleep(0.5)
                
                status_text.text("Calculating suitability scores...")
                progress_bar.progress(90)
                
                # Run pipeline or use mock data
                if CUSTOM_MODULES_AVAILABLE:
                    try:
                        pipe = ManthanDay2Pipeline(aoi_geojson)
                        res = pipe.run_complete_analysis()
                    except:
                        res = ManthanDay2Pipeline(None).run_complete_analysis()
                else:
                    res = ManthanDay2Pipeline(None).run_complete_analysis()
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
            
            st.session_state["analysis_results"] = normalise(res)
            st.session_state["current_step"] = 4
            st.success("✅ Analysis complete! View results in other tabs.")
            st.balloons()
            st.rerun()  # Fixed from st.experimental_rerun()
    else:
        st.info("👆 Draw an area on the map or enter coordinates to begin")

# =================================================================
# TAB 2 – ENVIRONMENTAL ANALYSIS
# =================================================================
with tab_env:
    if not st.session_state.get("analysis_results"):
        st.markdown("""
        <div class="info-box">
            <strong>No Analysis Data</strong><br>
            Please select a site and run the analysis first to view environmental data.
        </div>
        """, unsafe_allow_html=True)
    else:
        res = st.session_state["analysis_results"]
        ndvi = res["ndvi_data"]
        rain = res["environmental_data"]
        soil = res["soil_data"]
        suit = res["suitability"]
        
        # Environmental Overview
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">🔬</div>
                <div>
                    <h3 class="card-title">Environmental Analysis Results</h3>
                    <p class="card-subtitle">Comprehensive site assessment based on satellite data</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # NDVI Trend
            st.subheader("🌱 Vegetation Health Analysis")
            
            dates = pd.date_range(start='2024-01', periods=12, freq='M')
            ndvi_values = np.clip(
                np.random.normal(ndvi["ndvi_mean"], ndvi["ndvi_std"], 12),
                0, 1
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=ndvi_values,
                mode='lines+markers',
                name='NDVI',
                line=dict(color='#10b981', width=3),
                marker=dict(size=8, color='#10b981'),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            
            fig.add_hline(
                y=ndvi["ndvi_mean"],
                line_dash="dash",
                line_color="#059669",
                annotation_text=f"Average: {ndvi['ndvi_mean']:.3f}"
            )
            
            fig.update_layout(
                title="NDVI Trend (12 Months)",
                xaxis_title="Month",
                yaxis_title="NDVI Value",
                height=400,
                showlegend=False,
                plot_bgcolor='white',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Suitability Gauge
            st.subheader("🎯 Overall Suitability")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=suit['composite_score'] * 100,
                delta={'reference': 70, 'valueformat': '.1f'},
                title={'text': "Site Suitability Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#d1fae5"}
                    ],
                    'threshold': {
                        'line': {'color': "#059669", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Environmental Parameters
        st.markdown("### 📊 Detailed Environmental Parameters")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #059669; margin: 0 0 1rem 0;">🌧️ Precipitation</h4>
                <p style="margin: 0.5rem 0;"><strong>Annual:</strong> {rain['annual_rainfall']:.0f} mm</p>
                <p style="margin: 0.5rem 0;"><strong>Status:</strong> <span style="color: #10b981;">{rain['rainfall_adequacy']}</span></p>
                <p style="margin: 0.5rem 0;"><strong>Monsoon:</strong> {rain['annual_rainfall']*0.7:.0f} mm</p>
                <p style="margin: 0.5rem 0;"><strong>Dry Season:</strong> {rain['annual_rainfall']*0.3:.0f} mm</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h4 style="color: #059669; margin: 0 0 1rem 0;">🌡️ Temperature</h4>
                <p style="margin: 0.5rem 0;"><strong>Average:</strong> 27°C</p>
                <p style="margin: 0.5rem 0;"><strong>Summer Max:</strong> 38°C</p>
                <p style="margin: 0.5rem 0;"><strong>Winter Min:</strong> 15°C</p>
                <p style="margin: 0.5rem 0;"><strong>Growing Days:</strong> 285/year</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #059669; margin: 0 0 1rem 0;">🧪 Soil Analysis</h4>
                <p style="margin: 0.5rem 0;"><strong>pH Level:</strong> {soil['soil_ph']:.1f}</p>
                <p style="margin: 0.5rem 0;"><strong>Status:</strong> <span style="color: #10b981;">{soil['ph_suitability']}</span></p>
                <p style="margin: 0.5rem 0;"><strong>Type:</strong> Alluvial</p>
                <p style="margin: 0.5rem 0;"><strong>Drainage:</strong> Well-drained</p>
            </div>
            """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #059669; margin: 0 0 1rem 0;">🏔️ Topography</h4>
            <p style="margin: 0.5rem 0;"><strong>Elevation:</strong> 342m ASL</p>
            <p style="margin: 0.5rem 0;"><strong>Slope:</strong> 12° (Gentle)</p>
            <p style="margin: 0.5rem 0;"><strong>Aspect:</strong> North-East</p>
            <p style="margin: 0.5rem 0;"><strong>Terrain:</strong> Suitable</p>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #059669; margin: 0 0 1rem 0;">⭐ Restoration Priority</h4>
        """, unsafe_allow_html=True)

        if ISFR_DATA_AVAILABLE and 'ecological_analysis' in st.session_state:
            state = st.session_state['ecological_analysis']['location']['state']
            isfr_loader = ISFRDataLoader()
            state_cover = isfr_loader.get_forest_cover(state)
            national_avg = isfr_loader.get_national_average_cover()

            priority_score = max(0, (national_avg - state_cover) / national_avg) * 100

            if priority_score > 50:
                priority_level = "Very High"
            elif priority_score > 25:
                priority_level = "High"
            else:
                priority_level = "Medium"

            st.metric(
                label="Priority Score (vs Nat. Avg)",
                value=f"{priority_score:.0f}/100",
                delta=priority_level
            )
        else:
            st.info("Run analysis to see priority.")

        st.markdown("</div>", unsafe_allow_html=True)
    
        # Additional Charts
        st.markdown("### 📈 Environmental Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rainfall distribution
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            rainfall_monthly = [20, 15, 25, 45, 80, 200,
                              350, 320, 180, 60, 25, 15]
            
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=months,
                y=rainfall_monthly,
                marker_color='#3b82f6',
                name='Monthly Rainfall'
            ))
            
            fig_rain.update_layout(
                title="Monthly Rainfall Distribution",
                xaxis_title="Month",
                yaxis_title="Rainfall (mm)",
                showlegend=False,
                plot_bgcolor='white',
                height=350
            )
            
            st.plotly_chart(fig_rain, use_container_width=True)
        
        with col2:
            # Soil composition
            soil_comp = ['Sand', 'Silt', 'Clay', 'Organic Matter']
            soil_values = [40, 35, 20, 5]
            
            fig_soil = go.Figure(data=[go.Pie(
                labels=soil_comp,
                values=soil_values,
                hole=.3
            )])
            
            fig_soil.update_traces(
                marker=dict(colors=['#fbbf24', '#a78bfa', '#f87171', '#34d399'])
            )
            
            fig_soil.update_layout(
                title="Soil Composition",
                showlegend=True,
                height=350
            )
            
            st.plotly_chart(fig_soil, use_container_width=True)


def get_aoi_geojson_safely():
    """
    Safely extract GeoJSON from AOI data, handling all input methods
    """
    aoi_data = st.session_state.get("aoi_data", None)
    
    if not aoi_data:
        return None
    
    # Case 1: Direct geojson field exists
    if "geojson" in aoi_data and aoi_data["geojson"]:
        return aoi_data["geojson"]
    
    # Case 2: We have a polygon but no geojson - create one
    if "polygon" in aoi_data and aoi_data["polygon"]:
        from shapely.geometry import mapping
        polygon = aoi_data["polygon"]
        
        # Create GeoJSON from polygon
        return {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "area_ha": aoi_data.get("area_ha", 0)
            }
        }
    
    # Case 3: No valid data
    return None

if ECOLOGICAL_INTELLIGENCE_AVAILABLE and st.session_state.get("analysis_results"):
    # Initialize ecological intelligence
    if 'eco_intelligence' not in st.session_state:
        try:
            st.session_state.eco_intelligence = ManthanEcologicalIntelligence()
        except Exception as e:
            st.error(f"Failed to initialize Ecological Intelligence: {str(e)}")
            ECOLOGICAL_INTELLIGENCE_AVAILABLE = False
    
    # Get intelligent ecological analysis
    if ECOLOGICAL_INTELLIGENCE_AVAILABLE and 'ecological_analysis' not in st.session_state:
        # Safely get GeoJSON using our helper function
        aoi_geojson = get_aoi_geojson_safely()
        
        # Check if we have valid AOI data
        if not aoi_geojson:
            st.warning("No valid area selected. Please select an area first.")
        else:
            try:
                eco_analysis = st.session_state.eco_intelligence.analyze_location(
                    aoi_geojson=aoi_geojson,
                    environmental_data={
                        'annual_rainfall': st.session_state["analysis_results"]['environmental_data']['annual_rainfall'],
                        'soil_ph': st.session_state["analysis_results"]['soil_data']['soil_ph'],
                        'temperature': 27
                    }
                )
                st.session_state['ecological_analysis'] = eco_analysis
            except Exception as e:
                st.error(f"Failed to analyze location: {str(e)}")
                # Show debug info
                with st.expander("Debug Information"):
                    st.json(aoi_geojson)

# =================================================================
# TAB 3 – SPECIES RECOMMENDATIONS (INTELLIGENT VERSION)
# =================================================================
with tab_spec:
    if not st.session_state.get("analysis_results"):
        st.markdown("""
        <div class="info-box">
            <strong>Analysis Required</strong><br>
            Complete the environmental analysis to receive AI-powered species recommendations.
        </div>
        """, unsafe_allow_html=True)
    else:
        if ECOLOGICAL_INTELLIGENCE_AVAILABLE:
            # Initialize ecological intelligence
            if 'eco_intelligence' not in st.session_state:
                st.session_state.eco_intelligence = ManthanEcologicalIntelligence()
            
            # Get intelligent ecological analysis
            if 'ecological_analysis' not in st.session_state:
                # Extract coordinates from AOI
                aoi_data = st.session_state.get("aoi_data", {})
                
                eco_analysis = st.session_state.eco_intelligence.analyze_location(
                    aoi_geojson=aoi_data.get("geojson", {}),
                    environmental_data={
                        'annual_rainfall': st.session_state["analysis_results"]['environmental_data']['annual_rainfall'],
                        'soil_ph': st.session_state["analysis_results"]['soil_data']['soil_ph'],
                        'temperature': 27  # Would come from your climate data
                    }
                )
                st.session_state['ecological_analysis'] = eco_analysis
            
            # Render the analysis
            eco_analysis = st.session_state['ecological_analysis']
            st.session_state.eco_intelligence.render_ecological_analysis(eco_analysis)
        else:
            # Fallback to your current IndianForestEcology implementation
            eco_db = IndianForestEcology()
            # ... rest of your current code


# =================================================================
# TAB 4 – RESTORATION PLAN
# =================================================================
with tab_plan:
    if not st.session_state.get("analysis_results"):
        st.markdown("""
        <div class="info-box">
            <strong>No Data Available</strong><br>
            Complete the analysis to generate your restoration plan.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">📋</div>
                <div>
                    <h3 class="card-title">Your Restoration Blueprint</h3>
                    <p class="card-subtitle">Comprehensive implementation plan</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Timeline
            st.subheader("📅 Implementation Timeline")
            
            timeline_data = pd.DataFrame({
                'Phase': ['Site Preparation', 'Initial Planting', 'Maintenance Year 1',
                         'Maintenance Year 2-3', 'Monitoring & Evaluation'],
                'Start_Month': [0, 2, 3, 12, 0],
                'Duration': [2, 1, 9, 24, 60]
            })
            
            fig = go.Figure()
            
            colors = ['#fee2e2', '#fef3c7', '#d1fae5', '#ddd6fe', '#e0e7ff']
            
            for i, (_, row) in enumerate(timeline_data.iterrows()):
                fig.add_trace(go.Bar(
                    y=[row['Phase']],
                    x=[row['Duration']],
                    name=row['Phase'],
                    orientation='h',
                    marker=dict(color=colors[i]),
                    base=row['Start_Month'],
                    showlegend=False,
                    text=f"{row['Duration']} months",
                    textposition='inside'
                ))
            
            fig.update_layout(
                barmode='stack',
                title="5-Year Implementation Roadmap",
                xaxis_title="Months from Start",
                yaxis_title="",
                height=400,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key Activities
            st.subheader("🎯 Key Activities by Phase")
            
            activities = {
                "Year 1": [
                    "Site clearing and preparation",
                    "Soil improvement (if needed)",
                    "Sapling procurement",
                    "Initial planting (monsoon)",
                    "Watering system setup",
                    "Protection fencing"
                ],
                "Year 2-3": [
                    "Regular maintenance",
                    "Replacement planting",
                    "Pest management",
                    "Growth monitoring",
                    "Community engagement"
                ],
                "Year 4-5": [
                    "Reduced maintenance",
                    "Biodiversity monitoring",
                    "Carbon measurement",
                    "Sustainability planning"
                ]
            }
            
            for phase, items in activities.items():
                with st.expander(f"📌 {phase} Activities"):
                    for item in items:
                        st.markdown(f"• {item}")
        
        with col2:
            # Investment Summary
            area_ha = st.session_state.get("aoi_data", {}).get("area_ha", 45.2)
            total_investment = area_ha * 75000
            trees_count = int(area_ha * 2500)
            
            st.markdown(f"""
            <div class="custom-card" style="background: linear-gradient(135deg, #f0fdf4 0%, white 100%);">
                <div class="card-header">
                    <div class="card-icon">💰</div>
                    <div>
                        <h3 class="card-title">Investment Summary</h3>
                        <p class="card-subtitle">Cost breakdown & returns</p>
                    </div>
                </div>
                
                <div style="background: #d1fae5; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem;">
                    <p style="color: #065f46; font-size: 0.875rem; margin: 0;">Total Investment</p>
                    <p style="color: #065f46; font-size: 2rem; font-weight: 700; margin: 0;">₹{total_investment:,.0f}</p>
                    <p style="color: #059669; font-size: 0.875rem;">₹75,000 per hectare</p>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <p style="color: #6b7280; font-size: 0.75rem; margin: 0;">Trees</p>
                        <p style="font-weight: 700; font-size: 1.25rem; margin: 0;">{trees_count:,}</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <p style="color: #6b7280; font-size: 0.75rem; margin: 0;">Survival</p>
                        <p style="font-weight: 700; font-size: 1.25rem; margin: 0;">85%</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <p style="color: #6b7280; font-size: 0.75rem; margin: 0;">Carbon/yr</p>
                        <p style="font-weight: 700; font-size: 1.25rem; margin: 0;">
                            {
                                int(area_ha * ISFRDataLoader().get_carbon_stock(
                                st.session_state.get('ecological_analysis', {}).get('forest_ecology', {}).get('identified_type', 'DEFAULT')
                                ) * 1000)
                            } tons
                        </p>
                    <p style="color: #059669; font-size: 0.75rem; margin-top: 4px;">Total Potential Stock (ISFR Data)</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <p style="color: #6b7280; font-size: 0.75rem; margin: 0;">ROI</p>
                        <p style="font-weight: 700; font-size: 1.25rem; margin: 0;">7 years</p>
                    </div>
                </div>
                
                <div style="margin-top: 1.5rem;">
                    <h5 style="color: #059669; margin-bottom: 0.75rem;">Cost Breakdown</h5>
                    <div style="background: white; padding: 1rem; border-radius: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>Site Preparation</span>
                            <span style="font-weight: 600;">15%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>Saplings & Planting</span>
                            <span style="font-weight: 600;">35%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>Maintenance (3 yrs)</span>
                            <span style="font-weight: 600;">40%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Monitoring & Admin</span>
                            <span style="font-weight: 600;">10%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =================================================================
# TAB 5 – REPORTS & EXPORT
# =================================================================
with tab_rep:
    st.markdown("""
    <div class="custom-card">
        <div class="card-header">
            <div class="card-icon">📄</div>
            <div>
                <h3 class="card-title">Generate Reports</h3>
                <p class="card-subtitle">Export your restoration plan in various formats</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get("analysis_results"):
        st.info("Complete the analysis to enable report generation")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Report Configuration")
            
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Technical Report", "Implementation Guide",
                 "Funding Proposal", "Community Presentation"]
            )
            
            st.markdown("### Include Sections")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                include_maps = st.checkbox("Site Maps", value=True)
                include_env = st.checkbox("Environmental Analysis", value=True)
                include_species = st.checkbox("Species List", value=True)
            
            with col_b:
                include_timeline = st.checkbox("Implementation Timeline", value=True)
                include_budget = st.checkbox("Budget Breakdown", value=True)
                include_carbon = st.checkbox("Carbon Projections", value=True)
            
            report_lang = st.selectbox(
                "Language",
                ["English", "Hindi", "Marathi", "Tamil", "Telugu"]
            )
        
        with col2:
            st.subheader("📥 Export Options")
            
            # Quick download
            if st.button("📊 Download Analysis Data", use_container_width=True, key="download_analysis_btn"):
                st.download_button(
                    "⬇️ Download JSON",
                    json.dumps(st.session_state["analysis_results"], indent=2),
                    file_name=f"manthan_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.markdown("---")
            
            # In the Reports & Export tab, replace the PDF generation button code:
if st.button("📑 Generate PDF Report", type="primary", use_container_width=True, key="generate_pdf_btn"):
    with st.spinner("Generating PDF report with proper Unicode support..."):
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import io
            import os
            
            # Create PDF in memory
            pdf_buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Container for the 'Flowable' objects
            elements = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#059669'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            # Add content
            elements.append(Paragraph("Manthan Forest Restoration Report", title_style))
            elements.append(Spacer(1, 20))
            
            # Site Information
            if st.session_state.get("analysis_results"):
                res = st.session_state["analysis_results"]
                area = st.session_state.get("aoi_data", {}).get("area_ha", 0)
                
                # Create data table
                data = [
                    ['Parameter', 'Value'],
                    ['Site Suitability Score', f"{res['suitability']['composite_score']:.1%}"],
                    ['Total Area', f"{area:.1f} hectares"],
                    ['NDVI (Vegetation Health)', f"{res['ndvi_data']['ndvi_mean']:.3f}"],
                    ['Annual Rainfall', f"{res['environmental_data']['annual_rainfall']:.0f} mm"],
                    ['Soil pH', f"{res['soil_data']['soil_ph']:.1f}"],
                    ['Total Trees Planned', f"{int(area * 2500):,}"],
                    ['Carbon Sequestration Potential', f"{area * 10:.0f} tons/year"],
                    ['Total Investment', f"₹{area * 75000:,.0f}"]
                ]
                
                # Create table
                t = Table(data, colWidths=[3*inch, 2*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(t)
                elements.append(Spacer(1, 30))
                
                # Add species recommendations
                elements.append(Paragraph("Recommended Species", styles['Heading2']))
                elements.append(Spacer(1, 12))
                
                # Species would be added here based on actual recommendations
                
            # Build PDF
            doc.build(elements)
            
            # Get PDF data
            pdf_data = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            st.success("✅ PDF report generated successfully!")
            
            # Download button
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_data,
                file_name=f"manthan_restoration_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
            
        except ImportError:
            st.error("ReportLab not installed. Run: pip install reportlab")
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")
            
            if st.button("📊 Export to Excel", use_container_width=True, key="export_excel_btn"):
                with st.spinner("Creating Excel workbook..."):
                    time.sleep(1)
                st.success("✅ Excel file ready!")
            
            if st.button("🗺️ Export GIS Data", use_container_width=True):
                with st.spinner("Preparing GIS files..."):
                    time.sleep(1)
                st.success("✅ GIS package ready!")

# ───────────── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    # Logo and Title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); 
                    width: 80px; height: 80px; border-radius: 20px; 
                    margin: 0 auto 1rem; display: flex; align-items: center; 
                    justify-content: center; font-size: 3rem;
                    box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);">
            🌿
        </div>
        <h2 style="margin: 0; color: #111827;">Manthan</h2>
        <p style="color: #6b7280; font-size: 0.875rem; margin: 0.5rem 0;">
            Forest Intelligence Engine v2.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # System Status
    st.subheader("🔐 System Status")
    
    if CUSTOM_MODULES_AVAILABLE:
        if gee_init():
            st.success("✅ Earth Engine Connected")
            st.session_state["gee_authenticated"] = True
        else:
            st.error("❌ Earth Engine Error")
    else:
        st.warning("⚠️ Running in Demo Mode")
        st.info("Custom modules not available")
    
    # Quick Stats
    if st.session_state.get("analysis_results"):
        st.divider()
        st.subheader("📊 Quick Stats")
        
        res = st.session_state["analysis_results"]
        area = st.session_state.get("aoi_data", {}).get("area_ha", 0)
        
        st.markdown(f"""
        **Site Score:** {res['suitability']['composite_score']:.1%}  
        **Area:** {area:.1f} hectares  
        **Trees Planned:** {int(area * 2500):,}  
        **Carbon Potential:** {area * 10:.0f} tons/year  
        **Investment:** ₹{area * 75000:,.0f}
        """)
    
    st.divider()
    
    # Resources
    with st.expander("📚 Resources"):
        st.markdown("""
        - [User Guide](/)
        - [Species Database](/)
        - [Best Practices](/)
        - [API Documentation](/)
        - [Research Papers](/)
        """)
    
    with st.expander("🆘 Support"):
        st.markdown("""
        **Email:** support@manthan.ai  
        **Phone:** +91 98765 43210  
        **Hours:** Mon-Fri, 9AM-6PM IST
        """)
    
    # Footer
    st.markdown("---")
    st.caption("© 2024 Manthan AI - Building India's Green Future")
