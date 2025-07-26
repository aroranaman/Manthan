"""
Manthan: Regenerative Landscape Planner – Streamlit dashboard
Verified 2025-07-24
"""

# ───────────────────────── PATH FIX ────────────────────────────
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # …/Manthan
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# ───────────────────────────────────────────────────────────────

import streamlit as st
import ee, folium, geopandas as gpd, pandas as pd, numpy as np
from folium.plugins import Draw
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from datetime import datetime

from utils.gee_auth import gee_init
from scripts.day2_pipeline import ManthanDay2Pipeline

# ─────────────── PAGE CONFIG & CSS ─────────────────────────────
st.set_page_config(page_title="🌱 Manthan", page_icon="🌱",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .main-header{font-size:2.3rem;font-weight:bold;color:#2E7D2E;text-align:center;margin-bottom:1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────── KEY-NORMALISER (pipeline → UI) ──────────────────
def normalise(res: dict) -> dict:
    res.setdefault("ndvi_data",          res.get("ndvi",     {}))
    res.setdefault("environmental_data", res.get("rainfall", {}))
    res.setdefault("soil_data",          res.get("soil",     {}))
    return res

# ───────────── SESSION DEFAULTS ────────────────────────────────
for k, v in {
    "gee_authenticated": False,
    "aoi_data": None,
    "analysis_results": None,
}.items():
    st.session_state.setdefault(k, v)

# ───────────── SIDEBAR – AUTH ──────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/280x90/2E7D2E/FFFFFF?text=Manthan",
             caption="Forest Intelligence Engine")
    st.header("🔐 Earth Engine")
    if gee_init():
        st.success("Connected")
        st.session_state["gee_authenticated"] = True
    else:
        st.error("Authentication failed")
        st.stop()

# ───────────── TITLE ───────────────────────────────────────────
st.markdown('<p class="main-header">🌱 Manthan – Regenerative Landscape Planner</p>',
            unsafe_allow_html=True)

# ───────────── TABS ────────────────────────────────────────────
tab_site, tab_env, tab_spec, tab_rep = st.tabs(
    ["🗺️ Site", "📊 Analysis", "🌿 Species", "📋 Reports"]
)

# =================================================================
# TAB 1 – SITE SELECTION
# =================================================================
with tab_site:
    st.subheader("Draw or enter an Area-of-Interest")

    # 1 Folium map with draw toolbar
    fmap = folium.Map(location=[22.5, 79], zoom_start=5, tiles="OpenStreetMap")
    Draw(export=True,
         draw_options={"polyline":False,"circle":False,"marker":False,"circlemarker":False},
         edit_options={"edit":True}).add_to(fmap)

    draw_out = st_folium(fmap, height=480, width=700,
                         returned_objects=["last_active_drawing"])
    drawn = draw_out.get("last_active_drawing")

    # 2 Manual-coordinate entry
    with st.expander("📐 Create AOI from coordinates"):
        c1, c2 = st.columns(2)
        with c1:
            lat_min = st.number_input("Lat min", 20.0)
            lat_max = st.number_input("Lat max", 21.0)
        with c2:
            lon_min = st.number_input("Lon min", 78.0)
            lon_max = st.number_input("Lon max", 79.0)
        if st.button("➕ Add AOI from coords"):
            if lat_min < lat_max and lon_min < lon_max:
                coords = [
                    [lon_min, lat_min], [lon_max, lat_min],
                    [lon_max, lat_max], [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
                polygon = Polygon(coords)
                gdf = gpd.GeoDataFrame({"id":[1]}, geometry=[polygon], crs="EPSG:4326")
                st.session_state["aoi_data"] = {
                    "polygon": polygon,
                    "area_ha": gdf.to_crs("EPSG:3857").geometry.area.iloc[0]/10000,
                    "geojson": {
                        "type":"Feature","geometry":{"type":"Polygon","coordinates":[coords]},
                        "properties":{}},
                }
                st.success("AOI stored.")
            else:
                st.error("Min values must be < max values.")

    # 3 Accept drawn AOI
    if drawn and drawn["geometry"]["type"] in ("Polygon", "Rectangle"):
        if st.button("Use drawn AOI"):
            st.session_state["aoi_data"] = {
                "polygon": Polygon(drawn["geometry"]["coordinates"][0]),
                "area_ha": None,
                "geojson": drawn,
            }
            st.success("AOI stored.")

    # 4 Restoration method
    st.selectbox("🌳 Restoration method (affects species later)",
                 ["Auto-Detect","Miyawaki Dense Forest","Agroforestry",
                  "Eco-Tourism Forest"], key="restoration_method")

    # 5 Run pipeline
    if st.session_state.get("aoi_data"):
        if st.button("🔍 Run Day-2 analysis"):
            aoi_geojson = st.session_state["aoi_data"]["geojson"]
            with st.spinner("Running Day-2 pipeline…"):
                pipe = ManthanDay2Pipeline(aoi_geojson)
                res  = pipe.run_complete_analysis()
            st.session_state["analysis_results"] = normalise(res)
            st.success("Analysis finished – see other tabs.")
    else:
        st.info("Draw or add coordinates to enable analysis.")

# =================================================================
# TAB 2 – ENVIRONMENTAL ANALYSIS
# =================================================================
with tab_env:
    if not st.session_state.get("analysis_results"):
        st.info("Run an analysis first.")
    else:
        res  = st.session_state["analysis_results"]
        ndvi = res["ndvi_data"]
        rain = res["environmental_data"]
        soil = res["soil_data"]
        suit = res["suitability"]

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("🌱 NDVI", f"{ndvi['ndvi_mean']:.3f}", delta=f"±{ndvi['ndvi_std']:.3f}")
        k2.metric("🌧️ Rainfall", f"{rain['annual_rainfall']:.0f} mm",
                  delta=rain["rainfall_adequacy"])
        k3.metric("🌱 Soil pH", f"{soil['soil_ph']:.1f}", delta=soil["ph_suitability"])
        k4.metric("🎯 Suitability", f"{suit['composite_score']:.3f}",
                  delta=f"Grade {suit['suitability_grade']}")

        st.subheader("NDVI distribution")
        vals = np.clip(np.random.normal(ndvi["ndvi_mean"], ndvi["ndvi_std"], 1000), -1, 1)
        st.line_chart(pd.DataFrame({"NDVI":vals}))

# =================================================================
# TAB 3 – SPECIES (placeholder)
# =================================================================
with tab_spec:
    st.info("Species recommendation engine coming soon.")

# =================================================================
# TAB 4 – REPORTS (quick download)
# =================================================================
with tab_rep:
    if not st.session_state.get("analysis_results"):
        st.info("Run an analysis first.")
    else:
        st.download_button("⬇️ Download results JSON",
                           json.dumps(st.session_state["analysis_results"], indent=2),
                           file_name="manthan_results.json",
                           mime="application/json")
