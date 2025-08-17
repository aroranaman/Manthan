# FILE: src/app/intelligent_app.py
# FINAL VERSION: Fully integrated with advanced AI engine and real model outputs.

import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import plotly.graph_objects as go
import pandas as pd
import torch
import asyncio
import os
import time

# --- Robust Path Fix ---
import sys
from pathlib import Path
try:
    project_root = Path(__file__).resolve().parents[2]
except IndexError:
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Fix ---

from src.core.geospatial_handler import GeospatialDataHandler
from src.core.inference_pipeline import RegenerationInferencePipeline
from src.models.unet_segmenter import UNet
from src.core.ai_processing_engine import AIProcessingEngine 

st.set_page_config(page_title="Manthan AI", layout="wide", page_icon="🌿")

# --- Initialize Core Components & Session State ---
@st.cache_resource
def initialize_system():
    """Initializes and caches the core backend components."""
    handler = GeospatialDataHandler()
    
    unet_model_path = project_root / "saved_models" / "trained_unet_segmenter.pth"
    if not unet_model_path.parent.exists():
        unet_model_path.parent.mkdir(parents=True)
    if not unet_model_path.exists():
        st.warning(f"U-Net model not found. A dummy model will be created.")
        dummy_unet = UNet(in_channels=15, num_classes=5)
        dummy_unet.save(str(unet_model_path))
    pipeline = RegenerationInferencePipeline(model_path=str(unet_model_path))

    # Initialize the new AI Processing Engine
    ai_engine = AIProcessingEngine()
    
    return handler, pipeline, ai_engine

handler, inference_pipeline, ai_engine = initialize_system()

for key in ['aoi_geojson', 'suitability_map', 'advanced_analysis']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Main App UI ---
st.title("Manthan: AI-Powered Land Restoration Framework")
st.markdown("An AI-first platform to transform forest restoration from guesswork to data-driven precision.")

# --- TABS FOR STAKEHOLDER JOURNEYS ---
tab1, tab2, tab3, tab4 = st.tabs(["🧑‍🌾 Farmer / Landowner", "🤝 NGO / Community", "🏛️ Policymaker", "⚙️ System & Deployment"])

with tab1:
    st.header("Your Land, Your Livelihood: An AI-Powered Guide")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🗺️ Step 1: Define Your Land Parcel")
        m = folium.Map(location=[22.5, 79], zoom_start=5)
        Draw(export=True, position='topleft', draw_options={"rectangle": True, "polygon": True}).add_to(m)
        map_data = st_folium(m, height=450, width=700, key="folium_map")
        if map_data and map_data.get("last_active_drawing"):
            st.session_state.aoi_geojson = {"geometry": map_data["last_active_drawing"]['geometry']}
            st.success("Land parcel defined. Proceed to Step 2.")
    with col2:
        st.subheader("...or Enter Coordinates")
        with st.expander("Enter Bounding Box"):
            lat_min = st.number_input("Min Latitude", value=17.40, format="%.4f")
            lon_min = st.number_input("Min Longitude", value=78.40, format="%.4f")
            lat_max = st.number_input("Max Latitude", value=17.41, format="%.4f")
            lon_max = st.number_input("Max Longitude", value=78.41, format="%.4f")
            if st.button("Create Parcel from Coordinates", key="create_parcel_btn"):
                if lat_min < lat_max and lon_min < lon_max:
                    coords = [[[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]]
                    st.session_state.aoi_geojson = {"geometry": {"type": "Polygon", "coordinates": coords}}
                    st.success("Land parcel created. Proceed to Step 2.")
                else:
                    st.error("Min values must be less than max values.")
    st.markdown("---")
    st.subheader("🚀 Step 2: Run Full AI Analysis")
    if st.session_state.aoi_geojson:
        st.info("Land parcel is defined. Ready for full AI analysis.")
        if st.button("Generate My Restoration Plan", type="primary", key="generate_plan_btn"):
            with st.spinner("Running full AI pipeline... This may take a moment."):
                try:
                    patch, error_message = handler.get_multispectral_patch(st.session_state.aoi_geojson)
                    if patch is not None:
                        patch = patch.astype(np.float32)
                        st.session_state.suitability_map = inference_pipeline.predict(patch)
                        
                        # --- CORRECTED: Run Advanced AI Processing and store results ---
                        st.session_state.advanced_analysis = ai_engine.run_full_analysis(patch, st.session_state.aoi_geojson)
                        
                    else:
                        st.error(f"Failed to fetch data: {error_message}")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please define your land parcel to run the analysis.")

    if st.session_state.advanced_analysis is not None:
        st.markdown("---")
        st.header("✅ AI Assessment Complete: Your Restoration Blueprint")
        
        analysis = st.session_state.advanced_analysis
        
        # --- Display Advanced AI Outputs ---
        st.subheader("🧠 Intelligent Site Assessment")
        
        # Intervention Suitability
        st.markdown("**Intervention Suitability Scores (0-100)**")
        scores = analysis.get('intervention_suitability_scores', {})
        if scores:
            cols = st.columns(len(scores))
            for i, (intervention, score) in enumerate(scores.items()):
                cols[i].metric(intervention, f"{score:.1f}")
        else:
            st.warning("Suitability scores could not be generated.")

        st.markdown("---")

        # Risk Assessment
        st.subheader("ρίск Assessment")
        risks = analysis.get('risk_assessment', {})
        if risks:
            risk_df = pd.DataFrame(risks.items(), columns=['Risk Factor', 'Level'])
            risk_df['Level'] = risk_df['Level'].apply(lambda x: "High" if x > 0.66 else "Medium" if x > 0.33 else "Low")
            st.dataframe(risk_df, use_container_width=True)
        else:
            st.warning("Risk assessment could not be generated.")
        
        st.markdown("---")

        # Water Balance
        st.subheader("💧 Water Balance Analysis")
        water_balance = analysis.get('water_balance_analysis', {})
        if water_balance:
            wb_df = pd.DataFrame([water_balance])
            st.bar_chart(wb_df.T, use_container_width=True)
            st.caption("Monthly water surplus/deficit (in mm), critical for irrigation planning.")
        else:
            st.warning("Water balance analysis could not be generated.")

        st.markdown("---")

        # Regulatory Compliance
        st.subheader("📜 Regulatory Compliance Check")
        compliance = analysis.get('regulatory_compliance_check', {})
        if compliance:
            for check, status in compliance.items():
                st.markdown(f"- **{check}**: {status}")
        else:
            st.warning("Regulatory compliance check could not be completed.")
        
        st.success("This plan provides a comprehensive, multi-model AI analysis for your restoration project.")

with tab2:
    st.header("Coming Soon: NGO & Community Dashboard")

with tab3:
    st.header("Coming Soon: Policy & Planning Dashboard")

with tab4:
    st.header("Coming Soon: System & Deployment Dashboard")

# --- Footer ---
st.markdown("---")
st.markdown("**Manthan AI** - Transforming landscape restoration through artificial intelligence.")