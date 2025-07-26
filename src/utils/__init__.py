# src/utils/__init__.py
"""
Manthan Utils Package
Contains authentication, processing, and analysis utilities.
"""

# Import the actual functions from gee_auth module
from .gee_auth import gee_init, streamlit_authenticate

# Make these functions available when importing from utils
__all__ = [
    'gee_init',
    'streamlit_authenticate'
]
