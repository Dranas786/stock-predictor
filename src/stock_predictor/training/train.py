from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

from stock_predictor.assets.prices import raw_prices
from stock_predictor.assets.features import price_features
from stock_predictor.config import MLFLOW_TRACKING_URI


def make_training_frame(price_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tiny training dataset from an input feature DataFrame.

    This keeps the training logic decoupled from ingestion/orchestration.
    Dagster can pass `price_features` directly, and this function just prepares
    labels and drops invalid rows.
    """
    df = price_features_df.copy()

    # Target: will tomorrow's close be higher than today's close?
    df["target_up"] = (df["close_price"].shift(-1) > df["close_price"]).astype(int)

    # Drop rows where features/target are undefined (first diff is NaN, last target is NaN)
    df = df.dropna().reset_index(drop=True)

    return df


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time series friendly split: first chunk = train, last chunk = validation.
    (We do NOT shuffle time series.)
    """
    cutoff = max(1, int(len(df) * train_ratio))
    train_df = df.iloc[:cutoff].copy()
    val_df = df.iloc[cutoff:].copy()
    return train_df, val_df


def build_logreg_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Logistic regression with scaling (common strong baseline).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def build_rf_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Random forest baseline (does not need scaling).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """
    Evaluate on validation data with simple, explainable metrics.
    """
    preds = model.predict(X)
    return {
        "val_accuracy": float(accuracy_score(y, preds)),
        "val_f1": float(f1_score(y, preds)),
    }


def main(price_features_df: pd.DataFrame | None = None) -> None:
    """
    Train models and log to MLflow.

    If `price_features_df` is provided, we use it (Dagster path).
    If not, we create it from local assets (standalone/dev path).
    """
    # 1) Tell MLflow where the tracking server is
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("stock-predictor")

    # 2) Get features (Dagster passes them; standalone mode builds them)
    if price_features_df is None:
        price_features_df = price_features(raw_prices())

    df = make_training_frame(price_features_df)

    feature_cols = ["close_price", "daily_change"]
    target_col = "target_up"

    train_df, val_df = time_split(df, train_ratio=0.8)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # 3) Candidate models (simple bake-off)
    candidates: list[tuple[str, Pipeline]] = [
        ("logreg", build_logreg_pipeline(feature_cols)),
        ("random_forest", build_rf_pipeline(feature_cols)),
    ]

    best_name = None
    best_metric = -1.0
    best_metrics: dict[str, float] = {}

    # 4) Train + log each candidate as its own MLflow run
    for name, pipe in candidates:
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)

            metrics = evaluate(pipe, X_val, y_val)
            mlflow.log_metrics(metrics)

            mlflow.log_param("model_type", name)
            mlflow.log_param("feature_cols", ",".join(feature_cols))
            mlflow.log_param("n_train_rows", len(train_df))
            mlflow.log_param("n_val_rows", len(val_df))

            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                registered_model_name="stock-predictor-model",
            )

            score = metrics.get("val_f1", 0.0)
            if score > best_metric:
                best_metric = score
                best_name = name
                best_metrics = metrics

    print("Best model:", best_name)
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()