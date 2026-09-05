import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://forecast:forecast_secret@localhost:5432/demand_forecast",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

KAGGLE_DATASET = "rossmann-store-sales"

FORECAST_HORIZONS = [7, 14, 30]
