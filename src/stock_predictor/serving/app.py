from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_predictor.serving.load_model import load_latest_registered_model, predict


app = FastAPI(title="stock-predictor", version="0.1.0")

_model = None


class PredictRequest(BaseModel):
    close_price: float
    daily_change: float


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = load_latest_registered_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(req: PredictRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        return predict(_model, req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
