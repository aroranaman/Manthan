"""
Google Earth Engine Authentication Helper for Manthan
===================================================
Handles GEE authentication with fallback methods for development and production.
"""

import ee
import os
import streamlit as st
from pathlib import Path

class GEEAuthenticator:
    """Handles Google Earth Engine authentication."""
    
    def __init__(self, project_id: str = 'manthan-466509'):
        self.project_id = project_id
        self.service_account_path = 'config/gee-service-account.json'
        
    def authenticate(self) -> bool:
        """
        Authenticate with Google Earth Engine using multiple methods.
        
        Returns:
            bool: True if authentication successful
        """
        try:
            # Method 1: Try service account authentication
            if self._try_service_account():
                return True
                
            # Method 2: Try user authentication
            if self._try_user_auth():
                return True
                
            # Method 3: Try default credentials
            if self._try_default_auth():
                return True
                
            return False
            
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return False
    
    def _try_service_account(self) -> bool:
        """Try service account authentication."""
        try:
            if os.path.exists(self.service_account_path):
                credentials = ee.ServiceAccountCredentials(
                    email=None, 
                    key_file=self.service_account_path
                )
                ee.Initialize(credentials=credentials, project=self.project_id)
                return True
        except Exception:
            pass
        return False
    
    def _try_user_auth(self) -> bool:
        """Try user authentication."""
        try:
            ee.Initialize(project=self.project_id)
            # Test with a simple operation
            ee.Number(1).getInfo()
            return True
        except Exception:
            pass
        return False
    
    def _try_default_auth(self) -> bool:
        """Try default authentication."""
        try:
            ee.Initialize()
            return True
        except Exception:
            pass
        return False

def streamlit_authenticate() -> bool:
    """Streamlit-specific authentication with caching."""
    
    if 'gee_authenticated' not in st.session_state:
        with st.spinner("🔐 Connecting to Google Earth Engine..."):
            auth = GEEAuthenticator()
            success = auth.authenticate()
            st.session_state['gee_authenticated'] = success
            
            if success:
                st.success("✅ Google Earth Engine connected successfully!")
            else:
                st.error("❌ Google Earth Engine authentication failed")
                st.info("Please run 'earthengine authenticate' in your terminal")
                
    return st.session_state.get('gee_authenticated', False)

# Test function
if __name__ == "__main__":
    auth = GEEAuthenticator()
    if auth.authenticate():
        print("✅ Authentication successful!")
        
        # Test with simple operation
        test_number = ee.Number(42)
        result = test_number.getInfo()
        print(f"Test operation result: {result}")
    else:
        print("❌ Authentication failed")
