# Demand Forecasting & Replenishment Platform

End-to-end demand forecasting system that predicts product sales across stores for 7, 14, and 30 day horizons, and generates inventory replenishment recommendations.

## Features

- **Multi-model forecasting**: Seasonal Naive, Linear Regression, Random Forest, XGBoost/LightGBM, Prophet, LSTM, Temporal Fusion Transformer
- **Temporal backtesting**: Walk-forward validation with MAE, RMSE, WAPE, and forecast bias
- **Data enrichment**: Historical weather, public holidays, calendar features, promotions
- **Replenishment engine**: Safety stock, reorder quantities, stockout risk assessment
- **REST API**: FastAPI endpoints for forecasts and replenishment recommendations
- **Dashboard**: Interactive Streamlit dashboard with KPIs, forecasts, and model comparison
- **MLOps**: MLflow experiment tracking, model registry, Airflow orchestration

## Tech Stack

Python, PyTorch, XGBoost, LightGBM, Prophet, scikit-learn, FastAPI, Streamlit, PostgreSQL, MLflow, Airflow, Docker

## Setup

```bash
# Clone the repository
git clone https://github.com/SpiritGitHub/demand-forecasting-platform.git
cd demand-forecasting-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start infrastructure (PostgreSQL, MLflow)
docker-compose up -d

# Run data ingestion
python -m src.data.ingestion

# Run data enrichment
python -m src.data.enrichment
```

## Project Structure

```
├── src/
│   ├── data/           # Ingestion, validation, enrichment
│   ├── features/       # Feature engineering
│   ├── models/         # All forecasting models
│   ├── evaluation/     # Metrics and backtesting
│   └── replenishment/  # Inventory recommendation engine
├── api/                # FastAPI application
├── dashboard/          # Streamlit dashboard
├── airflow/            # DAGs for orchestration
├── monitoring/         # Data and model drift detection
├── notebooks/          # Exploratory analysis
├── tests/              # Unit and integration tests
└── data/               # Raw, processed, and feature data
```

## Architecture

```
Raw Data → ETL → PostgreSQL → Validation → Feature Engineering
    → Training (7 models) → MLflow → Model Registry
    → FastAPI → Dashboard → Monitoring
```

## Dataset

Based on [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle), enriched with:
- Historical weather data (Open-Meteo API)
- German public holidays
- Calendar features
