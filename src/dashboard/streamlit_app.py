"""
Manthan: Regenerative Forest Intelligence Engine
Dashboard Application with Google Earth Engine Integration
========================================================
"""

import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gee_auth import streamlit_authenticate
from utils.aoi_tools import AOIProcessor
from data_ingestion.gee_sentinel2 import GEESentinel2Processor
from models.suitability_engine import SuitabilityEngine

# Page configuration
st.set_page_config(
    page_title="🌱 Manthan: Forest Intelligence Engine",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e7b1e;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e7b1e 0%, #2d5a2d 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Header section
    st.markdown('<div class="main-header">🌱 Manthan: Forest Intelligence Engine</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Forest Restoration Planning for India</div>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    if not st.session_state.get('gee_authenticated', False):
        render_welcome_screen()
        return
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Site Selection", 
        "📊 Environmental Analysis", 
        "🌿 Species Recommendations",
        "📈 Analysis Results",
        "📋 Export & Reports"
    ])
    
    with tab1:
        site_selection_interface()
    
    with tab2:
        environmental_analysis_interface()
    
    with tab3:
        species_recommendations_interface()
    
    with tab4:
        analysis_results_interface()
    
    with tab5:
        export_reports_interface()

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    default_values = {
        'gee_authenticated': False,
        'aoi_data': None,
        'analysis_results': None,
        'species_recommendations': None,
        'ndvi_data': None,
        'processing_status': 'ready'
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    """Render sidebar with authentication and controls."""
    with st.sidebar:
        st.image("https://via.placeholder.com/280x80/1e7b1e/ffffff?text=MANTHAN", 
                caption="Regenerative Forest Intelligence")
        
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **🔐 Connect** to Google Earth Engine
        2. **🗺️ Select** your restoration site  
        3. **🔍 Analyze** environmental conditions
        4. **🌿 Get** AI species recommendations
        5. **📄 Export** restoration plan
        """)
        
        # Authentication section
        st.markdown("### 🔐 Authentication Status")
        if streamlit_authenticate():
            st.success("✅ Google Earth Engine Connected")
            st.session_state['gee_authenticated'] = True
        else:
            st.error("❌ Authentication Required")
            st.markdown("""
            **Setup Instructions:**
            1. Run `earthengine authenticate` in terminal
            2. Follow authentication prompts
            3. Refresh this page
            """)
            st.session_state['gee_authenticated'] = False
            return
        
        # Project status
        st.markdown("### 📊 Project Status")
        status_items = [
            ("Site Selected", st.session_state.get('aoi_data') is not None),
            ("Analysis Complete", st.session_state.get('analysis_results') is not None),
            ("Species Recommended", st.session_state.get('species_recommendations') is not None)
        ]
        
        for item, status in status_items:
            if status:
                st.success(f"✅ {item}")
            else:
                st.info(f"⏳ {item}")

def render_welcome_screen():
    """Render welcome screen when not authenticated."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🔐 Authentication Required</h3>
        <p>To use Manthan's forest intelligence capabilities, please authenticate with Google Earth Engine first.</p>
        <p><strong>Setup Instructions:</strong></p>
        <ol>
        <li>Open terminal/command prompt</li>
        <li>Run: <code>earthengine authenticate</code></li>
        <li>Follow the authentication prompts</li>
        <li>Refresh this page</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

def site_selection_interface():
    """Site selection interface with interactive map."""
    st.header("🗺️ Site Selection")
    st.markdown("Select your restoration site anywhere in India using the interactive map below.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create interactive map
        Map = geemap.Map(
            center=[20.5937, 78.9629],  # Center of India
            zoom=5,
            height="500px"
        )
        
        # Add base layers
        Map.add_basemap("SATELLITE")
        
        # Add India administrative boundary
        try:
            india = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
                ee.Filter.eq('ADM0_NAME', 'India')
            )
            Map.addLayer(
                india, 
                {'color': '#FF0000', 'width': 2, 'fillOpacity': 0}, 
                'India Boundary',
                False
            )
        except:
            pass
        
        # Display map
        map_data = Map.to_streamlit(height=500)
    
    with col2:
        st.subheader("🎯 Selection Method")
        
        # AOI selection options
        selection_method = st.selectbox(
            "Choose Selection Method:",
            ["Manual Coordinates", "Draw on Map", "Upload GeoJSON"],
            help="Select how you want to define your restoration area"
        )
        
        if selection_method == "Manual Coordinates":
            st.markdown("**Enter Bounding Box Coordinates:**")
            
            with st.form("coordinate_form"):
                lat_min = st.number_input("Latitude Min", value=18.5, min_value=6.0, max_value=38.0, step=0.001, format="%.3f")
                lat_max = st.number_input("Latitude Max", value=18.6, min_value=6.0, max_value=38.0, step=0.001, format="%.3f")
                lon_min = st.number_input("Longitude Min", value=73.8, min_value=68.0, max_value=98.0, step=0.001, format="%.3f")
                lon_max = st.number_input("Longitude Max", value=73.9, min_value=68.0, max_value=98.0, step=0.001, format="%.3f")
                
                submitted = st.form_submit_button("Create AOI", type="primary")
                
                if submitted:
                    create_aoi_from_coordinates(lat_min, lat_max, lon_min, lon_max)
        
        # Restoration method selection
        st.subheader("🌳 Restoration Approach")
        restoration_method = st.selectbox(
            "Preferred Method:",
            ["Auto-Detect (Recommended)", "Miyawaki Dense Forest", "Agroforestry System", "Eco-Tourism Forest", "Basic Revegetation"],
            help="Select restoration approach or let AI choose optimal method"
        )
        st.session_state['restoration_method'] = restoration_method
        
        # Analysis trigger
        if st.session_state.get('aoi_data'):
            aoi_info = st.session_state['aoi_data']
            
            st.markdown("### 📊 Selected Area Info")
            st.info(f"""
            **Area:** {aoi_info.get('area_ha', 0):.1f} hectares ({aoi_info.get('area_ha', 0)/100:.2f} km²)
            
            **Status:** {aoi_info.get('validation_message', 'Unknown')}
            """)
            
            if st.button("🔍 Analyze This Site", type="primary", use_container_width=True):
                run_comprehensive_analysis()
        else:
            st.info("Please select an area to continue")

def create_aoi_from_coordinates(lat_min, lat_max, lon_min, lon_max):
    """Create AOI from coordinate inputs."""
    if lat_min >= lat_max:
        st.error("Latitude Min must be less than Latitude Max")
        return
    
    if lon_min >= lon_max:
        st.error("Longitude Min must be less than Longitude Max")
        return
    
    # Create coordinate list
    coords = [
        [lon_min, lat_min],  # Southwest
        [lon_max, lat_min],  # Southeast
        [lon_max, lat_max],  # Northeast
        [lon_min, lat_max],  # Northwest
        [lon_min, lat_min]   # Close polygon
    ]
    
    # Process AOI
    processor = AOIProcessor()
    aoi_data = processor.create_aoi_from_coords(coords)
    
    if aoi_data.get('is_valid', False):
        st.session_state['aoi_data'] = aoi_data
        st.success(f"✅ AOI created successfully! Area: {aoi_data['area_ha']:.1f} hectares")
        st.rerun()
    else:
        st.error(f"❌ {aoi_data.get('validation_message', 'AOI creation failed')}")

def run_comprehensive_analysis():
    """Run comprehensive environmental analysis."""
    if not st.session_state.get('aoi_data'):
        st.error("No AOI selected")
        return
    
    aoi_data = st.session_state['aoi_data']
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize processors
        s2_processor = GEESentinel2Processor()
        suitability_engine = SuitabilityEngine()
        
        # Step 1: NDVI Analysis (40% progress)
        status_text.text("🛰️ Analyzing vegetation with Sentinel-2...")
        progress_bar.progress(0.2)
        
        ndvi_results = s2_processor.calculate_ndvi(
            aoi_data['ee_geometry'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        
        progress_bar.progress(0.4)
        st.session_state['ndvi_data'] = ndvi_results
        
        # Step 2: Simulate environmental data (60% progress)
        status_text.text("🌧️ Processing rainfall and soil data...")
        progress_bar.progress(0.6)
        
        # Simulate rainfall and soil data (replace with actual APIs later)
        rainfall_annual = np.random.normal(1100, 200)  # mm
        soil_ph = np.random.normal(6.8, 0.3)
        
        progress_bar.progress(0.8)
        
        # Step 3: Suitability analysis (80% progress)
        status_text.text("🧠 Computing AI suitability assessment...")
        
        assessment = suitability_engine.get_detailed_assessment(
            ndvi_results['ndvi_mean'],
            rainfall_annual,
            soil_ph,
            aoi_data['area_ha']
        )
        
        # Step 4: Species recommendations (100% progress)
        status_text.text("🌿 Generating species recommendations...")
        
        species_recommendations = suitability_engine.recommend_species(
            assessment['restoration_approach'],
            soil_ph,
            rainfall_annual
        )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Store results
        analysis_results = {
            **assessment,
            'ndvi_data': ndvi_results,
            'environmental_data': {
                'rainfall_annual': rainfall_annual,
                'soil_ph': soil_ph
            }
        }
        
        st.session_state['analysis_results'] = analysis_results
        st.session_state['species_recommendations'] = species_recommendations
        st.session_state['processing_status'] = 'complete'
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("🎉 Analysis completed successfully! Check other tabs for detailed results.")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Analysis failed: {str(e)}")

def environmental_analysis_interface():
    """Environmental analysis results interface."""
    st.header("📊 Environmental Analysis")
    
    if not st.session_state.get('analysis_results'):
        st.info("Please select a site and run analysis first.")
        return
    
    results = st.session_state['analysis_results']
    ndvi_data = results['ndvi_data']
    env_data = results['environmental_data']
    
    # Key metrics overview
    st.subheader("🎯 Key Environmental Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="🌱 NDVI",
            value=f"{ndvi_data['ndvi_mean']:.3f}",
            delta=f"±{ndvi_data['ndvi_std']:.3f}",
            help="Normalized Difference Vegetation Index"
        )
        st.markdown(f"**{ndvi_data['vegetation_coverage']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="🌧️ Annual Rainfall",
            value=f"{env_data['rainfall_annual']:.0f} mm",
            delta="Adequate" if env_data['rainfall_annual'] > 800 else "Low",
            help="Estimated annual precipitation"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="🌱 Soil pH",
            value=f"{env_data['soil_ph']:.1f}",
            delta="Optimal" if 6.0 <= env_data['soil_ph'] <= 7.5 else "Suboptimal",
            help="Soil acidity/alkalinity level"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="🎯 Suitability Score",
            value=f"{results['composite_score']:.3f}",
            delta=f"Grade {results['suitability_grade']}",
            help="Overall restoration suitability"
        )
        st.markdown(f"**{results['restoration_approach']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 NDVI Distribution")
        
        # Create NDVI histogram (simulated)
        ndvi_values = np.random.normal(ndvi_data['ndvi_mean'], ndvi_data['ndvi_std'], 1000)
        ndvi_values = np.clip(ndvi_values, -1, 1)  # Clip to valid NDVI range
        
        fig = px.histogram(
            x=ndvi_values,
            nbins=30,
            title="NDVI Value Distribution",
            labels={'x': 'NDVI', 'y': 'Frequency'}
        )
        fig.update_traces(marker_color='green', marker_opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌧️ Seasonal Rainfall Pattern")
        
        # Create seasonal rainfall chart (simulated)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_rain = [50, 30, 20, 40, 80, 150, 200, 180, 120, 90, 60, 40]
        
        fig = px.bar(
            x=months,
            y=monthly_rain,
            title="Monthly Rainfall Distribution",
            labels={'x': 'Month', 'y': 'Rainfall (mm)'}
        )
        fig.update_traces(marker_color='blue', marker_opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    # Factor analysis
    st.subheader("🔍 Factor Analysis")
    factor_analysis = results['factor_analysis']
    
    for factor_name, analysis in factor_analysis.items():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Rating badge
            rating = analysis['rating']
            color = {'Excellent': 'green', 'Good': 'blue', 'Fair': 'orange', 'Poor': 'red'}.get(rating, 'gray')
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
            <h4>{factor_name.replace('_', ' ').title()}</h4>
            <span style="background-color: {color}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px;">
            {rating}
            </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Value:** {analysis['value']}")
            st.markdown(f"**Assessment:** {analysis['description']}")

def species_recommendations_interface():
    """Species recommendations interface."""
    st.header("🌿 AI-Powered Species Recommendations")
    
    if not st.session_state.get('species_recommendations'):
        st.info("Please complete environmental analysis first.")
        return
    
    results = st.session_state['analysis_results']
    recommendations = st.session_state['species_recommendations']
    
    # Restoration strategy overview
    st.markdown(f"""
    <div class="success-box">
    <h3>🎯 Recommended Strategy: {results['restoration_approach']}</h3>
    <p><strong>Success Probability:</strong> {results['success_probability']}%</p>
    <p><strong>Suitability Grade:</strong> {results['suitability_grade']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Implementation metrics
    st.subheader("📋 Implementation Overview")
    metrics = results['implementation_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌳 Trees Needed", f"{metrics['total_trees_needed']:,}")
    with col2:
        st.metric("🌿 Species Count", metrics['recommended_species_count'])
    with col3:
        st.metric("⏳ Timeline", f"{metrics['establishment_timeline_years']} years")
    with col4:
        st.metric("💰 Est. Cost", f"₹{metrics['estimated_cost_inr']:,}")
    
    # Species by layer/category
    st.subheader("🌲 Recommended Species by Category")
    
    for category, species_list in recommendations.items():
        if species_list:
            st.markdown(f"### {category.replace('_', ' ').title()}")
            
            for i, species in enumerate(species_list):
                with st.expander(
                    f"🌿 {species['common_name']} ({species['scientific_name']})",
                    expanded=(i == 0)  # Expand first species
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Family:** {species['family']}")
                        st.markdown(f"**Growth Rate:** {species['growth_rate']}")
                        st.markdown(f"**Max Height:** {species['max_height']}m")
                        st.markdown(f"**Economic Value:** {species['economic_value']}")
                    
                    with col2:
                        st.markdown(f"**pH Range:** {species['soil_ph_min']:.1f} - {species['soil_ph_max']:.1f}")
                        st.markdown(f"**Rainfall:** {species['rainfall_min']}-{species['rainfall_max']}mm")
                        st.markdown(f"**Conservation Status:** {species['conservation_status']}")
                        st.markdown(f"**Native Regions:** {', '.join(species['native_regions'])}")
                    
                    st.markdown("**Ecological Functions:**")
                    for function in species['ecological_functions']:
                        st.markdown(f"• {function}")

def analysis_results_interface():
    """Comprehensive analysis results interface."""
    st.header("📈 Comprehensive Analysis Results")
    
    if not st.session_state.get('analysis_results'):
        st.info("Please complete analysis first.")
        return
    
    results = st.session_state['analysis_results']
    
    # Executive summary
    st.subheader("📋 Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Site Assessment:** {results['suitability_grade']} Grade ({results['composite_score']:.3f} score)
        
        **Recommended Approach:** {results['restoration_approach']}
        
        **Success Probability:** {results['success_probability']}%
        
        **Key Strengths:**
        """)
        
        # Identify strengths
        factor_analysis = results['factor_analysis']
        strengths = [f"• {factor.replace('_', ' ').title()}: {analysis['rating']}" 
                    for factor, analysis in factor_analysis.items() 
                    if analysis['rating'] in ['Excellent', 'Good']]
        
        for strength in strengths:
            st.markdown(strength)
    
    with col2:
        # Score visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results['composite_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Suitability Score"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.6], 'color': "yellow"},
                    {'range': [0.6, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations
    st.subheader("💡 Detailed Recommendations")
    
    recommendations = results['recommendations']
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    # Implementation timeline
    st.subheader("📅 Implementation Timeline")
    
    timeline_data = {
        'Phase': ['Site Preparation', 'Nursery Setup', 'Planting', 'Maintenance', 'Monitoring'],
        'Duration (months)': [2, 3, 6, 24, 36],
        'Activities': [
            'Soil testing, land clearing, water source development',
            'Seedling production, quality control',
            'Transplanting, initial watering, mulching',
            'Weeding, pruning, pest control, irrigation',
            'Growth tracking, survival assessment, adaptive management'
        ]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Timeline visualization
    fig = px.timeline(
        timeline_df,
        x_start=[0, 2, 5, 11, 11],
        x_end=[2, 5, 11, 35, 47],
        y='Phase',
        title="Implementation Timeline",
        color='Phase'
    )
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display timeline table
    st.dataframe(timeline_df, use_container_width=True)

def export_reports_interface():
    """Export and reports interface."""
    st.header("📋 Export & Reports")
    
    if not st.session_state.get('analysis_results'):
        st.info("Please complete analysis to generate reports.")
        return
    
    results = st.session_state['analysis_results']
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Available Reports")
        
        report_options = {
            "Site Analysis Report": "Comprehensive environmental assessment",
            "Species Planting Guide": "Detailed species recommendations with planting instructions",
            "Implementation Timeline": "Project schedule and milestone tracking",
            "Cost Estimation": "Budget breakdown and financial projections"
        }
        
        selected_reports = []
        for report_name, description in report_options.items():
            if st.checkbox(f"{report_name}", value=True):
                selected_reports.append(report_name)
                st.caption(description)
    
    with col2:
        st.subheader("🎨 Export Options")
        
        export_format = st.selectbox("Format", ["PDF", "Excel", "Word", "CSV"])
        include_maps = st.checkbox("Include maps and visualizations", True)
        include_photos = st.checkbox("Include species photographs", False)
        language = st.selectbox("Language", ["English", "Hindi"])
        
        st.subheader("📧 Delivery Options")
        email_delivery = st.checkbox("Email to project stakeholders")
        if email_delivery:
            st.text_input("Email addresses (comma-separated)")
    
    # Data export section
    st.subheader("📊 Raw Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📍 AOI GeoJSON", use_container_width=True):
            if st.session_state.get('aoi_data'):
                aoi_geojson = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(st.session_state['aoi_data']['polygon'].exterior.coords)]
                    },
                    "properties": {
                        "area_hectares": st.session_state['aoi_data']['area_ha'],
                        "analysis_date": datetime.now().isoformat()
                    }
                }
                
                st.download_button(
                    "Download AOI GeoJSON",
                    data=str(aoi_geojson),
                    file_name="manthan_aoi.geojson",
                    mime="application/geo+json"
                )
    
    with col2:
        if st.button("📈 Analysis CSV", use_container_width=True):
            # Create analysis summary CSV
            analysis_data = {
                'Metric': ['NDVI Mean', 'NDVI Std', 'Annual Rainfall', 'Soil pH', 'Suitability Score', 'Success Probability'],
                'Value': [
                    results['ndvi_data']['ndvi_mean'],
                    results['ndvi_data']['ndvi_std'],
                    results['environmental_data']['rainfall_annual'],
                    results['environmental_data']['soil_ph'],
                    results['composite_score'],
                    results['success_probability']
                ],
                'Unit': ['NDVI', 'NDVI', 'mm', 'pH', 'Score', '%']
            }
            
            df = pd.DataFrame(analysis_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                "Download Analysis CSV",
                data=csv_data,
                file_name="manthan_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🌿 Species List", use_container_width=True):
            if st.session_state.get('species_recommendations'):
                species_data = []
                
                for category, species_list in st.session_state['species_recommendations'].items():
                    for species in species_list:
                        species_data.append({
                            'Category': category.replace('_', ' ').title(),
                            'Scientific Name': species['scientific_name'],
                            'Common Name': species['common_name'],
                            'Family': species['family'],
                            'Growth Rate': species['growth_rate'],
                            'Max Height (m)': species['max_height'],
                            'Economic Value': species['economic_value']
                        })
                
                df = pd.DataFrame(species_data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    "Download Species CSV",
                    data=csv_data,
                    file_name="manthan_species.csv",
                    mime="text/csv"
                )
    
    # Generate comprehensive report
    if selected_reports:
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                # Simulate report generation
                import time
                time.sleep(2)
                
                # Create report content
                report_content = generate_report_content(results, selected_reports)
                
                st.success("✅ Report generated successfully!")
                
                st.download_button(
                    f"📥 Download {export_format} Report",
                    data=report_content,
                    file_name=f"manthan_report_{datetime.now().strftime('%Y%m%d_%H%M')}.{export_format.lower()}",
                    mime="application/octet-stream"
                )

def generate_report_content(results, selected_reports):
    """Generate report content based on selected options."""
    report_lines = [
        "# Manthan Forest Restoration Analysis Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        f"**Restoration Approach:** {results['restoration_approach']}",
        f"**Suitability Score:** {results['composite_score']:.3f} (Grade {results['suitability_grade']})",
        f"**Success Probability:** {results['success_probability']}%",
        "",
        "## Environmental Analysis",
        f"**NDVI:** {results['ndvi_data']['ndvi_mean']:.4f} ± {results['ndvi_data']['ndvi_std']:.4f}",
        f"**Vegetation Coverage:** {results['ndvi_data']['vegetation_coverage']}",
        f"**Annual Rainfall:** {results['environmental_data']['rainfall_annual']:.0f} mm",
        f"**Soil pH:** {results['environmental_data']['soil_ph']:.1f}",
        "",
        "## Implementation Metrics",
        f"**Trees Required:** {results['implementation_metrics']['total_trees_needed']:,}",
        f"**Species Diversity:** {results['implementation_metrics']['recommended_species_count']}",
        f"**Timeline:** {results['implementation_metrics']['establishment_timeline_years']} years",
        f"**Estimated Cost:** ₹{results['implementation_metrics']['estimated_cost_inr']:,}",
        "",
        "## Recommendations"
    ]
    
    for i, rec in enumerate(results['recommendations'], 1):
        report_lines.append(f"{i}. {rec}")
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    main()
