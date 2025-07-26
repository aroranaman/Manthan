import streamlit as st, geemap.foliumap as geemap, json, os, tempfile
from utils import auth, ndvi, rainfall, soil_ph, export, suitability

st.set_page_config(page_title="Suitability Scoring", layout="wide")
auth.gee_init()

# AOI selector
with st.sidebar:
    st.markdown("### Draw or upload AOI")
    aoi_geojson = geemap.draw_export()  # returns dict
    if st.button("Run Day 2 Pipeline") and aoi_geojson:
        with st.spinner("Processing…"):
            run_id = export.run_pipeline(aoi_geojson)  # wrapper around scripts
        st.success("Done! Check the Results tab.")

if aoi_geojson:
    m = geemap.Map(center=[20, 80], zoom=5)
    m.add_geojson(aoi_geojson, layer_name="AOI")
    st.components.v1.html(m._repr_html_(), height=600)
