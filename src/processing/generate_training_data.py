import ee
import json
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "manthan-466509"  # your working GEE project
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "data"

# Local outputs
DISTRICT_MAP_JSON = DATA_DIR / "district_mapping.json"
LOCAL_SAMPLES_JSON = DATA_DIR / "training_samples.json"

# Sampling sizes (keep small to avoid timeouts on free tier)
SAMPLES_PER_CLASS_LOCAL = 100   # ~300 total (Forest/Rural/Other)
SCALE_METERS = 100              # sampling scale in meters

# WorldCover -> 3 classes
# ESA WC classes: 10 Tree cover | 20 Shrub | 30 Grassland | 40 Cropland | 50 Built-up | 60 Barren | 70 Snow/Ice | 80 Open water | 90 Herb.wetland | 95 Mangroves | 100 Moss/Lichen
FROM_CLASSES = [10,  20,  30,  40,  50,   60,   70,  80,  90,  95,  100]
TO_CLASSES   = [ 2,   0,   0,   1,   0,    0,    0,   0,   0,   0,    0]  # 2=Forest, 1=Rural (cropland), 0=Other

# Optional: start a Drive export for a bigger set (off by default; can exceed free tier)
START_FULL_DRIVE_EXPORT = False
FULL_SAMPLES_PER_CLASS = 1000   # ~3000 total — may be too big for free tier


def safe_init():
    """Initialize Earth Engine, authenticate if needed."""
    print("Initializing Google Earth Engine...")
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def build_india_boundaries():
    """Return India Level-0 and Level-2 boundaries FeatureCollections."""
    print("Fetching administrative boundaries...")
    level0 = ee.FeatureCollection("FAO/GAUL/2015/level0") \
        .filter(ee.Filter.eq('ADM0_NAME', 'India'))
    level2 = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filterBounds(level0.geometry())
    return level0, level2


def save_district_mapping(level2_fc):
    """Create and save district -> id mapping locally (json)."""
    names = level2_fc.aggregate_array('ADM2_NAME').getInfo()
    unique = sorted(list(set(names)))
    mapping = {name: i for i, name in enumerate(unique)}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISTRICT_MAP_JSON, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved district mapping for {len(unique)} districts to {DISTRICT_MAP_JSON}")
    return mapping


def build_wc3():
    """Create a 3-class WorldCover image with band name 'wc3'."""
    print("Building WorldCover → 3-class image...")
    wc_map = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc3 = wc_map.remap(FROM_CLASSES, TO_CLASSES).rename("wc3")
    return wc3


def stratified_samples(wc3_img, region, per_class, scale=SCALE_METERS):
    """Return a FeatureCollection of stratified samples (with 'wc3' + 'rand' bands)."""
    print(f"Stratified sampling ~{per_class} per class...")
    # FIX: use a numeric seed and give the band a unique name
    rand = ee.Image.random(42).rename('rand')
    img = wc3_img.addBands(rand)

    samples = img.stratifiedSample(
        numPoints=per_class,
        classBand="wc3",
        region=region,
        scale=scale,
        geometries=True,      # include geometry so we can get lon/lat
        dropNulls=True,
        classValues=[0, 1, 2],
        classPoints=[per_class, per_class, per_class]
    )
    # Keep only the needed bands to reduce payload
    samples = samples.select(["wc3", "rand"])
    return samples


def attach_district(samples_fc, districts_fc):
    """Map over points and attach 'district_name' property."""
    def _attach(feature):
        d = districts_fc.filterBounds(feature.geometry()).first()
        name = ee.Algorithms.If(
            ee.Algorithms.IsEqual(d, None),
            "Unknown",
            ee.Feature(d).get("ADM2_NAME")
        )
        return feature.set({"district_name": name})

    return samples_fc.map(_attach)


def to_lightweight_points(samples_fc, limit_n=None):
    """
    Reduce properties/geometry to a small payload that can be safely getInfo'd on free tier.
    Returns a FeatureCollection of points with only {lon, lat, class, district_name}.
    """
    def _light(feature):
        geom = feature.geometry().centroid(1)
        coords = ee.List(geom.coordinates())
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        cls = ee.Number(feature.get('wc3'))
        dist = feature.get('district_name')
        return ee.Feature(geom, {
            "longitude": lon,
            "latitude": lat,
            "class": cls,
            "district_name": dist
        })

    fc = samples_fc.map(_light)
    if limit_n is not None:
        # sort by the random column we added earlier
        fc = fc.sort("rand").limit(limit_n)
    # ensure only minimal props are kept
    fc = fc.map(lambda f: ee.Feature(f).select(
        propertySelectors=["longitude", "latitude", "class", "district_name"]
    ))
    return fc


def save_locally_as_json(samples_fc, out_path: Path):
    """Fetch a small FeatureCollection to the client and save as JSON."""
    print("Fetching sample features locally...")
    features = samples_fc.getInfo()["features"]

    rows = []
    for f in features:
        props = f["properties"]
        rows.append({
            "longitude": props["longitude"],
            "latitude": props["latitude"],
            "class": int(props["class"]),            # 0=Other, 1=Rural, 2=Forest
            "district_name": props["district_name"]
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"✅ Local sample saved to {out_path} ({len(rows)} rows)")


def start_drive_export(samples_fc):
    """
    Optional: Export full samples as CSV to Google Drive (server-side).
    WARNING: Large exports can exceed free-tier limits.
    """
    export_props = ["wc3", "district_name", "rand"]
    task = ee.batch.Export.table.toDrive(
        collection=samples_fc.select(export_props),
        description="Manthan_TrainingSamples_Full",
        folder="manthan_training_patches",
        fileNamePrefix="training_samples_full",
        fileFormat="CSV"
    )
    task.start()
    print("🚚 Started Drive export for full dataset.")


def main():
    safe_init()

    # Admin boundaries
    india0, india2 = build_india_boundaries()
    _ = save_district_mapping(india2)

    # WorldCover → 3-class
    wc3 = build_wc3()

    # Stratified sampling over India
    samples_fc = stratified_samples(
        wc3_img=wc3,
        region=india0.geometry(),
        per_class=SAMPLES_PER_CLASS_LOCAL,
        scale=SCALE_METERS
    )

    # Attach district names (server-side)
    samples_with_district = attach_district(samples_fc, india2)

    # Make a lightweight subset for local save
    lightweight_local = to_lightweight_points(
        samples_with_district,
        limit_n=SAMPLES_PER_CLASS_LOCAL * 3
    )

    # Save locally as JSON (fast, safe on free tier)
    save_locally_as_json(lightweight_local, LOCAL_SAMPLES_JSON)

    # Optional: start a Drive export for a larger dataset (off by default)
    if START_FULL_DRIVE_EXPORT:
        full_samples = stratified_samples(
            wc3_img=wc3,
            region=india0.geometry(),
            per_class=FULL_SAMPLES_PER_CLASS,
            scale=SCALE_METERS
        )
        full_with_district = attach_district(full_samples, india2)
        start_drive_export(full_with_district)


if __name__ == "__main__":
    main()
