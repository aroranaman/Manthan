# FILE: scripts/train_forecasting_models.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import joblib
import sys


# --- Robust Path Fix ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Fix ---

# --- CONFIG ---
DATA_DIR = project_root / "src" / "data"
SAVED_MODEL_DIR = project_root / "saved_models"
SAVED_MODEL_DIR.mkdir(exist_ok=True)
TRAINING_DATA_PATH = DATA_DIR / "forecasting_training_data.csv"

def train():
    """Trains three separate LightGBM models for forecasting."""
    print("Loading forecasting training data...")
    df = pd.read_csv(TRAINING_DATA_PATH)

    features = [
        'NDVI_mean', 'total_precip_mean', 'LST_Day_mean', 
        'soil_organic_carbon_mean', 'percent_fruit_trees',
        'percent_medicinal_plants', 'percent_timber_trees'
    ]
    targets = [
        'time_to_sustainability_years', 
        'annual_foraging_income', 
        '10yr_carbon_sequestration'
    ]
    
    X = df[features]
    y = df[targets]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train a separate model for each target ---
    for target in targets:
        print(f"\n--- Training model for: {target} ---")
        
        y_train_target = y_train[target]
        y_test_target = y_test[target]
        
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train_target)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test_target, preds)
        print(f"Validation Mean Absolute Error: {mae:.4f}")
        
        # Save the trained model
        model_path = SAVED_MODEL_DIR / f"forecaster_{target}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train()