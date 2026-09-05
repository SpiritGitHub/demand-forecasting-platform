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

# Run feature engineering
python -m src.features.engineering

# Run training + backtesting (requires MLflow running)
python -m src.training.train
```

## Project Structure

```
├── src/
│   ├── config.py           # Paths, DB connection, constants
│   ├── data/
│   │   ├── ingestion.py    # Kaggle download, cleaning, PostgreSQL insert
│   │   ├── enrichment.py   # Weather, holidays, calendar features
│   │   └── validation.py   # Pandera schemas, date continuity checks
│   ├── features/
│   │   └── engineering.py  # Lags, rolling, promo, competition, interactions
│   ├── models/
│   │   ├── base.py         # Abstract BaseForecaster interface
│   │   ├── naive.py        # Seasonal Naive (baseline)
│   │   ├── linear.py       # Ridge regression
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── prophet_model.py  # Prophet (aggregate + per-store scaling)
│   │   └── lstm.py         # LSTM (PyTorch, per-store sequences)
│   ├── evaluation/
│   │   ├── metrics.py      # MAE, RMSE, WAPE, forecast bias
│   │   └── backtesting.py  # Walk-forward temporal cross-validation
│   └── training/
│       └── train.py        # Full pipeline: 7 models + MLflow logging
├── notebooks/
│   └── 01_exploration.ipynb  # EDA: distributions, STL, promo lift, outliers
├── api/                # FastAPI application (Phase 6)
├── dashboard/          # Streamlit dashboard (Phase 7)
├── airflow/            # DAGs for orchestration (Phase 8)
├── monitoring/         # Data and model drift detection (Phase 8)
├── tests/              # Unit and integration tests (Phase 9)
├── docs/
│   └── documentation.html  # Full technical docs (open in browser)
├── data/
│   ├── raw/            # Raw CSVs + cleaned parquet
│   ├── external/       # Cached weather data
│   ├── processed/      # Enriched dataset
│   └── features/       # ML-ready features
└── scripts/
    └── init_db.sql     # PostgreSQL schema
```

## Architecture

```
Raw Data → ETL → PostgreSQL → Validation → Feature Engineering
    → Training (7 models) → MLflow → Model Registry
    → FastAPI → Dashboard → Monitoring
```

## Feature Engineering

The `src/features/engineering.py` module generates ~30 ML features per row:

| Category | Features |
|----------|----------|
| Lag | `sales_lag_7`, `sales_lag_14`, `sales_lag_28` |
| Rolling | `rolling_mean_7/14/28`, `rolling_std_7`, `ewm_7` |
| Trend | `sales_trend` (short vs long moving average) |
| Promo | `promo_duration` (consecutive promo days) |
| Competition | `competition_distance_log`, `competition_open_months` |
| Store | `store_type_enc`, `assortment_enc`, `store_mean/median/std_sales` |
| Interactions | `promo_x_weekend`, `promo_x_holiday`, `holiday_x_store_type` |
| Day profile | `store_dow_mean_sales`, `store_dow_ratio` |

All lag/rolling features use `shift(1)` to prevent data leakage. Store aggregates accept a `train_end_date` parameter for safe backtesting.

## Models & Backtesting

7 models evaluated via **walk-forward temporal cross-validation** (3 folds, 14-day horizon):

| Model | Type | Key Parameters |
|-------|------|----------------|
| Seasonal Naive | Baseline | Repeat last week |
| Ridge | Linear | alpha=1.0 |
| Random Forest | Ensemble | 200 trees, depth 15 |
| XGBoost | Boosting | 300 rounds, lr=0.05 |
| LightGBM | Boosting | 300 rounds, 63 leaves |
| Prophet | Time series | Weekly + yearly seasonality |
| LSTM | Deep learning | 2 layers, hidden=64, seq=14 |

Metrics: **MAE**, **RMSE**, **WAPE** (Weighted Absolute Percentage Error), **Forecast Bias**.

Results logged to MLflow and saved to `data/results/backtesting_results.csv`.

## Dataset

Based on [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle), enriched with:
- Historical weather data (Open-Meteo API)
- German public holidays
- Calendar features

## Documentation

Open `docs/documentation.html` in your browser for the full technical documentation with syntax-highlighted code and detailed French explanations.
