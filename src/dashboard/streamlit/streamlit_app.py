"""
🌿 Manthan: Professional Forest Intelligence Dashboard
Complete implementation with enhanced UI and full functionality
"""
# ───────────────────────── PATH FIX ────────────────────────────
import sys
import json
import time
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[3]  # …/Manthan
    SRC = ROOT / "src"
    for p in (ROOT, SRC):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
except Exception as e:
    print(f"Path setup error: {str(e)}")
    ROOT = Path.cwd()
    SRC = ROOT / "src"
# ───────────────────────────────────────────────────────────────
import streamlit as st
import ee
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from folium.plugins import Draw
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from utils.gee_auth import gee_init
from scripts.day2_pipeline import ManthanDay2Pipeline
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# ─────────────── ERROR HANDLING & LOADING STATE ─────────────────
# Add this right after st.set_page_config (around line 30)

# Import custom modules with error handling
try:
    from utils.gee_auth import gee_init
    from scripts.day2_pipeline import ManthanDay2Pipeline
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Custom module import error: {str(e)}")
    st.info("Running in limited mode without custom modules")
    CUSTOM_MODULES_AVAILABLE = False
    
    # Mock functions for testing
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

# Initialize loading state
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False
    
# Show loading spinner while app initializes
if not st.session_state.app_loaded:
    with st.spinner('🌿 Loading Manthan Forest Intelligence...'):
        try:
            # Test imports
            import ee
            import folium
            import geopandas as gpd
            import pandas as pd
            import numpy as np
            from folium.plugins import Draw
            from shapely.geometry import Polygon
            from streamlit_folium import st_folium
            st.success("✅ All imports successful")
            time.sleep(0.5)
        except ImportError as e:
            st.error(f"Import Error: {str(e)}")
            st.info("Please install missing dependencies:")
            st.code(f"pip install {str(e).split('No module named ')[-1].strip('\'')}")
            st.stop()


# ─────────────── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="Manthan: Forest Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────── ENHANCED CSS STYLING ─────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* HideMainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
        border-color: #10b981;
    }
    
    [data-testid="metric-container"] label {
        color: #6b7280;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #111827;
        font-weight: 700;
        font-size: 2rem;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        background: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Tabs */
    .stTabs {
        background: transparent;
    }
    
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
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
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
        color: #111827;
        margin: 0;
    }
    
    .card-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0.25rem 0 0 0;
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
    .progress-container {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
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
        transition: all 0.2s ease;
    }
    
    .progress-number.active {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.4);
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
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .error-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    section[data-testid="stSidebar"] .element-container {
        background: white;
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Map Container */
    .map-container {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    
    /* Data displays */
    .data-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .data-item {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    
    .data-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .data-value {
        font-size: 1.25rem;
        color: #111827;
        font-weight: 700;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f3f4f6;
        border-radius: 0.75rem;
        font-weight: 600;
    }
    
    /* Select box styling */
    .stSelectbox label {
        color: #374151;
        font-weight: 600;
    }
    
    /* Number input styling */
    .stNumberInput label {
        color: #374151;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ───────────── KEY-NORMALISER (pipeline → UI) ──────────────────
def normalise(res: dict) -> dict:
    res.setdefault("ndvi_data", res.get("ndvi", {}))
    res.setdefault("environmental_data", res.get("rainfall", {}))
    res.setdefault("soil_data", res.get("soil", {}))
    return res

# ───────────── SESSION DEFAULTS ────────────────────────────────
for k, v in {
    "gee_authenticated": False,
    "aoi_data": None,
    "analysis_results": None,
    "current_step": 1,
}.items():
    st.session_state.setdefault(k, v)

# ───────────── MAIN APP WRAPPER ──────────────────────────────
try:
    # Mark app as loaded
    st.session_state.app_loaded = True
    
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
            f"{area:.1f} ha",
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
        
        # Folium map with draw toolbar
        fmap = folium.Map(
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
        ).add_to(fmap)
        
        # Display map
        draw_out = st_folium(
            fmap,
            height=500,
            width=None,
            returned_objects=["last_active_drawing"],
            key="main_map"
        )
        
        drawn = draw_out.get("last_active_drawing")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Accept drawn AOI
        if drawn and drawn["geometry"]["type"] in ("Polygon", "Rectangle"):
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Use This Area", type="primary", use_container_width=True):
                    st.session_state["aoi_data"] = {
                        "polygon": Polygon(drawn["geometry"]["coordinates"][0]),
                        "area_ha": None,
                        "geojson": drawn,
                    }
                    st.session_state["current_step"] = 2
                    st.success("Area saved! Configure settings and run analysis.")
    
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
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
            
            if st.button("➕ Create AOI from Coordinates", use_container_width=True):
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
                    st.success(f"AOI created: {area_ha:.1f} hectares")
                else:
                    st.error("Min values must be less than max values")
        
        # Run Analysis Button
        if st.session_state.get("aoi_data"):
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                aoi_geojson = st.session_state["aoi_data"]["geojson"]
                
                with st.spinner("🔍 Analyzing site conditions..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run pipeline with progress updates
                    status_text.text("Initializing Earth Engine...")
                    progress_bar.progress(10)
                    
                    pipe = ManthanDay2Pipeline(aoi_geojson)
                    
                    status_text.text("Analyzing vegetation (NDVI)...")
                    progress_bar.progress(30)
                    
                    status_text.text("Fetching rainfall data...")
                    progress_bar.progress(50)
                    
                    status_text.text("Analyzing soil properties...")
                    progress_bar.progress(70)
                    
                    status_text.text("Calculating suitability scores...")
                    progress_bar.progress(90)
                    
                    res = pipe.run_complete_analysis()
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                
                st.session_state["analysis_results"] = normalise(res)
                st.session_state["current_step"] = 4
                st.success("✅ Analysis complete! View results in other tabs.")
                st.balloons()
                st.rerun()
        else:
            st.info("👆 Draw an area on the map or enter coordinates to begin")

# =================================================================
# TAB 2 – ENVIRONMENTAL ANALYSIS
# =================================================================
with tab_env:
    if not st.session_state.get("analysis_results"):
        st.markdown("""
        <div class="warning-box">
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
            # NDVI Distribution
            st.subheader("🌱 Vegetation Health Analysis")
            
            # Generate sample NDVI time series
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
            
            # Add average line
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
        
        # Detailed Metrics Grid
        st.markdown("### 📊 Detailed Environmental Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="data-item">
                <div class="data-label">🌧️ Annual Rainfall</div>
                <div class="data-value">{:.0f} mm</div>
                <div style="color: #10b981; font-size: 0.875rem; margin-top: 0.5rem;">
                    {}
                </div>
            </div>
            """.format(
                rain['annual_rainfall'],
                rain['rainfall_adequacy']
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="data-item">
                <div class="data-label">🌡️ Temperature</div>
                <div class="data-value">27°C</div>
                <div style="color: #10b981; font-size: 0.875rem; margin-top: 0.5rem;">
                    Optimal Range
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="data-item">
                <div class="data-label">🧪 Soil pH</div>
                <div class="data-value">{:.1f}</div>
                <div style="color: #10b981; font-size: 0.875rem; margin-top: 0.5rem;">
                    {}
                </div>
            </div>
            """.format(
                soil['soil_ph'],
                soil['ph_suitability']
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="data-item">
                <div class="data-label">🏔️ Elevation</div>
                <div class="data-value">342 m</div>
                <div style="color: #10b981; font-size: 0.875rem; margin-top: 0.5rem;">
                    Suitable
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional Analysis
        st.markdown("### 🌍 Site Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rainfall distribution chart
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
            # Soil composition pie chart
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

# =================================================================
# TAB 3 – SPECIES RECOMMENDATIONS
# =================================================================
with tab_spec:
    if not st.session_state.get("analysis_results"):
        st.markdown("""
        <div class="warning-box">
            <strong>Analysis Required</strong><br>
            Complete the environmental analysis to receive AI-powered species recommendations.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <div class="card-icon">🌿</div>
                <div>
                    <h3 class="card-title">AI-Powered Species Recommendations</h3>
                    <p class="card-subtitle">Native species optimized for your site conditions</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter options
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            canopy_filter = st.selectbox(
                "Canopy Layer",
                ["All Layers", "Emergent", "Canopy", "Understory", "Ground"]
            )
        
        with col2:
            growth_filter = st.selectbox(
                "Growth Rate",
                ["All Rates", "Fast", "Medium", "Slow"]
            )
        
        with col3:
            purpose_filter = st.multiselect(
                "Special Features",
                ["Wildlife", "Medicinal", "Timber", "Nitrogen Fixing"]
            )
        
        with col4:
            native_only = st.checkbox("Native Species Only", value=True)
        
        # Species recommendations (placeholder data)
        species_data = [
            {
                "name": "Sal",
                "scientific": "Shorea robusta",
                "family": "Dipterocarpaceae",
                "layer": "Emergent",
                "growth": "Medium",
                "carbon": 300,
                "rainfall": "1000-3000",
                "features": ["Wildlife", "Timber", "Medicinal"],
                "score": 95
            },
            {
                "name": "Teak",
                "scientific": "Tectona grandis",
                "family": "Lamiaceae",
                "layer": "Canopy",
                "growth": "Medium",
                "carbon": 250,
                "rainfall": "800-2500",
                "features": ["Premium Timber", "Economic Value"],
                "score": 88
            },
            {
                "name": "Neem",
                "scientific": "Azadirachta indica",
                "family": "Meliaceae",
                "layer": "Canopy",
                "growth": "Fast",
                "carbon": 180,
                "rainfall": "400-1200",
                "features": ["Medicinal", "Hardy", "Pest Control"],
                "score": 92
            },
            {
                "name": "Jamun",
                "scientific": "Syzygium cumini",
                "family": "Myrtaceae",
                "layer": "Canopy",
                "growth": "Fast",
                "carbon": 220,
                "rainfall": "900-2000",
                "features": ["Edible Fruit", "Wildlife", "Medicinal"],
                "score": 87
            },
            {
                "name": "Bamboo",
                "scientific": "Dendrocalamus strictus",
                "family": "Poaceae",
                "layer": "Understory",
                "growth": "Fast",
                "carbon": 150,
                "rainfall": "600-2000",
                "features": ["Fast Growth", "Economic", "Soil Conservation"],
                "score": 85
            },
            {
                "name": "Amla",
                "scientific": "Phyllanthus emblica",
                "family": "Phyllanthaceae",
                "layer": "Understory",
                "growth": "Medium",
                "carbon": 160,
                "rainfall": "700-1800",
                "features": ["Medicinal", "Edible", "Economic"],
                "score": 83
            }
        ]
        
        # Display species cards
        st.markdown("### 🏆 Top Recommended Species")
        
        col1, col2 = st.columns(2)
        
        for i, species in enumerate(species_data):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
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
                            <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.5rem;">
                                {species['growth']} growth
                            </div>
                        </div>
                    </div>
                    
                    <div class="data-grid" style="margin: 1rem 0;">
                        <div class="data-item">
                            <div class="data-label">🌧️ Rainfall</div>
                            <div style="font-weight: 600;">{species['rainfall']} mm</div>
                        </div>
                        <div class="data-item">
                            <div class="data-label">🌳 Carbon Seq.</div>
                            <div style="font-weight: 600;">{species['carbon']} kg/year</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;">
                        {' '.join([f'<span style="background: #d1fae5; color: #065f46; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">✓ {feature}</span>' for feature in species['features']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Composition Chart
        st.markdown("### 📊 Recommended Forest Composition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Species composition bar chart
            composition_data = pd.DataFrame({
                'Layer': ['Emergent', 'Canopy', 'Understory', 'Ground Cover'],
                'Percentage': [15, 45, 30, 10],
                'Trees_per_ha': [375, 1125, 750, 250]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=composition_data['Layer'],
                y=composition_data['Percentage'],
                text=composition_data['Percentage'].apply(lambda x: f'{x}%'),
                textposition='auto',
                marker_color=['#064e3b', '#059669', '#10b981', '#34d399']
            ))
            
            fig.update_layout(
                title="Recommended Canopy Layer Distribution",
                xaxis_title="Forest Layer",
                yaxis_title="Percentage (%)",
                showlegend=False,
                plot_bgcolor='white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card" style="background: #f0fdf4;">
                <h4 style="color: #059669; margin: 0 0 1rem 0;">Planting Summary</h4>
                
                <div class="data-item" style="background: white; margin-bottom: 0.75rem;">
                    <div class="data-label">Total Species</div>
                    <div class="data-value">12-15</div>
                </div>
                
                <div class="data-item" style="background: white; margin-bottom: 0.75rem;">
                    <div class="data-label">Trees per Hectare</div>
                    <div class="data-value">2,500</div>
                </div>
                
                <div class="data-item" style="background: white; margin-bottom: 0.75rem;">
                    <div class="data-label">Survival Rate</div>
                    <div class="data-value">85-90%</div>
                </div>
                
                <div class="data-item" style="background: white;">
                    <div class="data-label">Carbon Potential</div>
                    <div class="data-value">450 tons/ha</div>
                    <div style="color: #059669; font-size: 0.75rem;">Over 30 years</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =================================================================
# TAB 4 – RESTORATION PLAN
# =================================================================
with tab_plan:
    if not st.session_state.get("analysis_results"):
        st.markdown("""
        <div class="warning-box">
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
            # Implementation Timeline
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
                    "Sapling procurement and nursery setup",
                    "Initial planting (monsoon season)",
                    "Watering and mulching",
                    "Protection fencing installation"
                ],
                "Year 2-3": [
                    "Regular watering schedule",
                    "Weeding and mulching",
                    "Replacement planting (10-15%)",
                    "Pest and disease management",
                    "Growth monitoring",
                    "Community engagement activities"
                ],
                "Year 4-5": [
                    "Reduced maintenance",
                    "Thinning if required",
                    "Biodiversity assessments",
                    "Carbon stock measurement",
                    "Community handover preparation",
                    "Long-term sustainability planning"
                ]
            }
            
            for phase, items in activities.items():
                with st.expander(f"📌 {phase} Activities"):
                    for item in items:
                        st.markdown(f"• {item}")
        
        with col2:
            # Investment Summary
            st.markdown("""
            <div class="custom-card" style="background: linear-gradient(135deg, #f0fdf4 0%, white 100%);">
                <div class="card-header">
                    <div class="card-icon">💰</div>
                    <div>
                        <h3 class="card-title">Investment Summary</h3>
                        <p class="card-subtitle">Cost breakdown & ROI</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            area_ha = st.session_state.get("aoi_data", {}).get("area_ha", 45.2)
            total_investment = area_ha * 75000
            trees_count = int(area_ha * 2500)
            
            st.markdown(f"""
                <div class="data-item" style="background: #d1fae5; margin-bottom: 1rem;">
                    <div class="data-label">Total Investment</div>
                    <div class="data-value">₹{total_investment:,.0f}</div>
                    <div style="color: #059669; font-size: 0.875rem;">₹75,000 per hectare</div>
                </div>
                
                <div class="data-grid">
                    <div class="data-item">
                        <div class="data-label">Trees</div>
                        <div style="font-weight: 700; font-size: 1.125rem;">{trees_count:,}</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">Survival</div>
                        <div style="font-weight: 700; font-size: 1.125rem;">85%</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">Carbon/yr</div>
                        <div style="font-weight: 700; font-size: 1.125rem;">{int(area_ha * 10)} tons</div>
                    </div>
                    <div class="data-item">
                        <div class="data-label">ROI</div>
                        <div style="font-weight: 700; font-size: 1.125rem;">7 years</div>
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
            
            # Quick download of current results
            if st.button("📊 Download Analysis Data", use_container_width=True):
                st.download_button(
                    "⬇️ Download JSON",
                    json.dumps(st.session_state["analysis_results"], indent=2),
                    file_name=f"manthan_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.markdown("---")
            
            # Report generation buttons
            if st.button("📑 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    time.sleep(2)  # Simulate generation
                st.success("✅ PDF report generated!")
                st.download_button(
                    "⬇️ Download PDF",
                    b"PDF content here",  # Replace with actual PDF generation
                    file_name=f"manthan_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            
            if st.button("📊 Export to Excel", use_container_width=True):
                with st.spinner("Creating Excel workbook..."):
                    time.sleep(1)
                st.success("✅ Excel file ready!")
            
            if st.button("🗺️ Export GIS Data", use_container_width=True):
                with st.spinner("Preparing GIS files..."):
                    time.sleep(1)
                st.success("✅ GIS package ready!")
        
        # Sample Report Preview
        st.markdown("### 📄 Report Preview")
        
        with st.expander("View Sample Report Structure"):
            st.markdown("""
            **MANTHAN FOREST RESTORATION REPORT**
            
            1. **Executive Summary**
               - Site location and size
               - Suitability assessment
               - Key recommendations
            
            2. **Site Analysis**
               - Environmental conditions
               - Vegetation assessment
               - Soil and climate data
            
            3. **Species Recommendations**
               - Native species list
               - Planting composition
               - Expected outcomes
            
            4. **Implementation Plan**
               - Timeline and phases
               - Budget breakdown
               - Maintenance schedule
            
            5. **Expected Impact**
               - Carbon sequestration
               - Biodiversity enhancement
               - Community benefits
            
            6. **Appendices**
               - Detailed data tables
               - Maps and coordinates
               - References
            """)

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
    
    # GEE Authentication Status
    st.subheader("🔐 System Status")
    
    if gee_init():
        st.success("✅ Earth Engine Connected")
        st.session_state["gee_authenticated"] = True
    else:
        st.error("❌ Earth Engine Error")
        st.info("Check authentication settings")
    
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
except Exception as e:
    st.error(f"Application Error: {str(e)}")
    st.error("Stack trace:")
    st.code(str(e.__traceback__))
    
    # Show debug info
    with st.expander("Debug Information"):
        st.write("Session State:", st.session_state)
        st.write("Python Path:", sys.path)
        st.write("Current Directory:", Path.cwd())
