"""Phase 4: Model comparison and champion selection."""

import logging

import mlflow
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, RESULTS_DIR

logger = logging.getLogger(__name__)

RANKING_METRIC = "wape"


def load_backtesting_results(path=None):
    if path is None:
        path = RESULTS_DIR / "backtesting_results.csv"
    return pd.read_csv(path)


def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae", "rmse", "wape", "bias"]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")

    summary = results.groupby("model").agg(**agg).reset_index()
    summary = summary.sort_values(f"{RANKING_METRIC}_mean")
    summary["rank"] = range(1, len(summary) + 1)
    return summary


def select_champion(summary: pd.DataFrame) -> str:
    return summary.sort_values(f"{RANKING_METRIC}_mean").iloc[0]["model"]


def compare_models(results_path=None):
    results = load_backtesting_results(results_path)
    summary = compute_summary(results)
    champion = select_champion(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    (RESULTS_DIR / "champion.txt").write_text(champion)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("demand-forecasting-phase4")

    with mlflow.start_run(run_name=f"champion-{champion}"):
        mlflow.log_param("champion_model", champion)
        mlflow.log_param("ranking_metric", RANKING_METRIC)

        row = summary[summary["model"] == champion].iloc[0]
        for m in ["mae", "rmse", "wape", "bias"]:
            mlflow.log_metric(f"champion_{m}_mean", row[f"{m}_mean"])
            mlflow.log_metric(f"champion_{m}_std", row[f"{m}_std"])

        mlflow.log_artifact(str(RESULTS_DIR / "model_comparison.csv"))
        mlflow.log_artifact(str(RESULTS_DIR / "champion.txt"))

    logger.info("Champion: %s (WAPE=%.4f)", champion, row["wape_mean"])
    return summary, champion


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
    )
    summary, champion = compare_models()
    print("\n=== Model Comparison (ranked by WAPE) ===")
    print(summary.to_string(index=False))
    print(f"\nChampion: {champion}")
