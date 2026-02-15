from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from stock_predictor.config import MLFLOW_TRACKING_URI


def load_latest_registered_model(model_name: str = "stock-predictor-model") -> mlflow.pyfunc.PyFuncModel:
    """
    Load the latest Production model from the MLflow Model Registry.

    If you haven't promoted anything to Production yet, this will fall back to Staging,
    and then to the latest version (best-effort) so the demo still works.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()

    # 1) Try Production
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if versions:
        uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(uri)

    # 2) Try Staging
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if versions:
        uri = f"models:/{model_name}/Staging"
        return mlflow.pyfunc.load_model(uri)

    # 3) Fallback: load latest version number (if stages not used)
    all_versions = client.search_model_versions(f"name='{model_name}'")
    if not all_versions:
        raise RuntimeError(
            f"No registered model found for '{model_name}'. "
            "Train a model first so MLflow registers it."
        )

    latest_version = max(int(v.version) for v in all_versions)
    uri = f"models:/{model_name}/{latest_version}"
    return mlflow.pyfunc.load_model(uri)


def predict(model: mlflow.pyfunc.PyFuncModel, features: dict[str, Any]) -> dict[str, Any]:
    """
    Run a single prediction.

    `features` should contain keys like: close_price, daily_change
    """
    df = pd.DataFrame([features])
    preds = model.predict(df)

    # mlflow.pyfunc returns a numpy array / pandas series depending on model flavor
    pred_value = float(preds[0])

    return {"prediction": pred_value}
