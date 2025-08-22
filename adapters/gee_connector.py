# adapters/gee_connector.py
import ee

# A simple flag to prevent re-initialization in a single run
_ee_initialized = False

def initialize_gee():
    """
    Initializes the Google Earth Engine API.
    
    This function ensures that GEE is initialized only once per session.
    It uses a simple module-level flag for tracking.
    """
    global _ee_initialized
    if _ee_initialized:
        print("INFO: GEE is already initialized.")
        return True
    
    try:
        # NOTE: You might need to add authentication logic here if not
        # handled by the environment (e.g., ee.Authenticate()).
        # For now, we assume the user is authenticated via the CLI.
        ee.Initialize(project='manthan-466509')
        
        # A simple test to confirm the connection works.
        ee.Number(1).getInfo()
        
        _ee_initialized = True
        print("✅ INFO: Google Earth Engine initialized successfully.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Google Earth Engine: {e}")
        _ee_initialized = False
        return False