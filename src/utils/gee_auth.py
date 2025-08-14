# utils/gee_auth.py - SIMPLIFIED VERSION
import ee
import streamlit as st

def gee_init():
    """Initialize Google Earth Engine."""
    if 'ee_initialized' in st.session_state:
        return True
    
    try:
        # Use the same project you authenticated with
        ee.Initialize(project='manthan-466509')
        
        # Test connection
        ee.Number(1).getInfo()
        st.session_state.ee_initialized = True
        return True
        
    except Exception as e:
        print(f"GEE Error: {e}")
        return False

def streamlit_authenticate():
    """Wrapper function for backward compatibility."""
    return gee_init()
