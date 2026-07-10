"""MLflow setup shared by all notebooks.

Team setup (DagsHub — free hosted MLflow server, shared by both teammates):
set these three environment variables before starting jupyter
(or put them in a .env file, which is gitignored):

    MLFLOW_TRACKING_URI      = https://dagshub.com/<owner>/walmart-sales-forecasting.mlflow
    MLFLOW_TRACKING_USERNAME = <your dagshub username>
    MLFLOW_TRACKING_PASSWORD = <your dagshub token>

Without them, everything falls back to a local sqlite db
(view with `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo
root) — fine for solo debugging.
"""
import os
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_MODEL_NAME = "walmart-best-model"


def setup_mlflow(experiment_name):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        # local fallback: sqlite (newer MLflow rejects the ./mlruns file store)
        uri = "sqlite:///" + (REPO_ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow -> {uri} | experiment: {experiment_name}")
