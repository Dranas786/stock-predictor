from dagster import asset
import pandas as pd
import mlflow

from stock_predictor.config import MLFLOW_TRACKING_URI
from stock_predictor.training.train import main as train_main


@asset
def trained_model(price_features: pd.DataFrame) -> str:
    """
    Train candidate models and log everything to MLflow.

    The `price_features` input exists so Dagster enforces the dependency:
    raw_prices -> price_features -> trained_model.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Reuse the existing training entrypoint
    train_main(price_features)

    return "Training complete (logged to MLflow)."
