import geopandas as gpd, rasterio, rasterio.features, json, numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

def classify_score(score_arr):
    classes = np.full(score_arr.shape, 0, dtype=np.uint8)
    classes[score_arr >= 0.65] = 3  # Miyawaki Dense
    classes[(score_arr >= 0.40) & (score_arr < 0.65)] = 2  # Agroforestry
    classes[score_arr < 0.40] = 1   # Basic Revegetation
    return classes

def raster_to_geojson(raster_arr, profile, dst):
    results = (
        {"properties": {"class": int(v)}, "geometry": s}
        for s, v in rasterio.features.shapes(raster_arr, transform=profile["transform"])
    )
    gdf = gpd.GeoDataFrame.from_features(results, crs=profile["crs"])
    gdf.to_file(dst, driver='GeoJSON')

def write_pdf(score_stats, map_png, dst):
    doc = SimpleDocTemplate(dst)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Manthan – Restoration Blueprint", styles['Title']),
        Paragraph(f"AOI area: {score_stats['area_km2']:.1f} km²", styles['Normal']),
        RLImage(map_png, width=500, height=350)
    ]
    doc.build(story)
