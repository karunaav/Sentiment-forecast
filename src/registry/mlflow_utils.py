from __future__ import annotations
import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "sentiment-forecast")

def get_client():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient(tracking_uri=MLFLOW_URI)

def start_run(ticker: str):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.start_run(run_name=f"{ticker}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")

def log_artifacts(params: dict, metrics: dict, model_path: str, ticker: str):
    with start_run(ticker):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
