# Manthan: Regenerative Landscape Planner

## AI-Powered Forest Intelligence Engine for India

Manthan is a next-generation ecosystem restoration platform that uses Google Earth Engine, satellite imagery, and AI to provide scientifically grounded forest restoration recommendations.

## Quick Setup

### 1. Environment Setup

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate manthan-gee
```

**Option B: Using pip**
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 2. Google Earth Engine Authentication

```bash
# First-time setup
earthengine authenticate

# Verify authentication
python -c "import ee; ee.Initialize(project='manthan-466509'); print('✅ GEE Connected')"
```

### 3. Launch Dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

## Project Structure

```
manthan/
├── src/
│   ├── dashboard/           # Streamlit interface
│   ├── data_ingestion/      # GEE data processors
│   ├── processing/          # Analysis algorithms
│   ├── models/              # AI/ML components
│   └── utils/               # Helper functions
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # AOI-clipped results
│   └── outputs/             # Reports & exports
└── config/                  # Configuration files
```

## Features

- 🗺️ Interactive AOI selection using Google Earth Engine
- 📊 Sentinel-2 NDVI analysis with cloud masking
- 🌧️ CHIRPS rainfall data integration
- 🌱 AI-powered native species recommendations
- 📋 Automated report generation

## Technology Stack

- **Satellite Data**: Google Earth Engine
- **Frontend**: Streamlit + Folium
- **Geospatial**: GeoPandas, Rasterio, Shapely
- **AI/ML**: Species recommendation engine with GBIF integration

## Getting Started

1. Select an Area of Interest (AOI) on the interactive map
2. Run environmental analysis (NDVI, rainfall, soil)
3. Get AI-powered species recommendations
4. Export restoration plans and reports

## Support

For setup issues or questions, refer to the documentation in the `docs/` folder.