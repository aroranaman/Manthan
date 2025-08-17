# config.py

"""
Central configuration file for the Manthan project.

Stores API keys, database credentials, file paths, and other settings.
It's recommended to use environment variables for sensitive data.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file (optional, for local development)
load_dotenv()

# --- Google Earth Engine Configuration ---
# Path to your GEE service account JSON key file.
# Set this as an environment variable: export GEE_SERVICE_ACCOUNT_KEY='/path/to/your/key.json'
GEE_SERVICE_ACCOUNT_KEY = os.getenv('GEE_SERVICE_ACCOUNT_KEY')
GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID', 'your-gcp-project-id')

# --- Database Configuration ---
# Credentials for connecting to the PostgreSQL/PostGIS database.
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'manthan_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD') # Always load passwords from env

# --- WhatsApp API Configuration (e.g., Twilio) ---
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER') # e.g., 'whatsapp:+14155238886'

# --- File and Directory Paths ---
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the directory where trained models are stored
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SPECIES_RECOMMENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'species_ncf_model.h5')
TREE_DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, 'tree_detection_model.pt')

# Path to the data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
INTEGRATED_SPECIES_DB_PATH = os.path.join(DATA_DIR, 'processed', 'integrated_species_database.csv')

# --- Validation and Checks ---
if not GEE_SERVICE_ACCOUNT_KEY:
    print("Warning: GEE_SERVICE_ACCOUNT_KEY environment variable is not set.")

if not DB_PASSWORD:
    print("Warning: DB_PASSWORD environment variable is not set. Database connection may fail.")