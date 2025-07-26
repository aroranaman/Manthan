# utils/gee_auth.py - CORRECTED VERSION for earthengine-api 1.6.0
import ee
import streamlit as st
import json
from google.oauth2.credentials import Credentials

def gee_init():
    """Initialize Google Earth Engine with proper 1.6.0 authentication."""
    if 'ee_initialized' in st.session_state:
        return True
    
    PROJECT_ID = "222841508645"  # Your project number (as string)
    
    try:
        if "EARTHENGINE_TOKEN" in st.secrets:
            # Parse the complete token
            token_data = json.loads(st.secrets["EARTHENGINE_TOKEN"])
            
            # Create proper OAuth2 credentials object
            credentials = Credentials(
                token=None,  # Access token (will be refreshed)
                refresh_token=token_data['refresh_token'],
                id_token=None,
                token_uri='https://oauth2.googleapis.com/token',
                client_id=token_data['client_id'],
                client_secret=token_data['client_secret'],
                scopes=token_data.get('scopes', [])
            )
            
            # Initialize with credentials object and project
            ee.Initialize(credentials=credentials, project=PROJECT_ID)
            
        else:
            # Fallback to persistent credentials with project
            ee.Initialize(credentials='persistent', project=PROJECT_ID)
        
        # Test the connection
        ee.Number(1).getInfo()
        st.session_state.ee_initialized = True
        return True
        
    except Exception as e:
        st.error(f"🚨 GEE Authentication failed: {str(e)}")
        return False

def streamlit_authenticate():
    """Wrapper function for backward compatibility."""
    return gee_init()
