import json
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

from preprocessing import add_features

MODEL_PATH = Path(__file__).with_name("house_price_model.joblib")
METRICS_PATH = Path(__file__).with_name("model_metrics.json")

app = FastAPI(
    title="California House Price API",
    description="API for predicting California housing prices using the trained model.",
    version="1.0",
)


class PredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income in tens of thousands")
    HouseAge: float = Field(..., description="Average house age in years")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Population in the block")
    AveOccup: float = Field(..., description="Average occupancy per household")
    Latitude: float = Field(..., description="Latitude coordinate")
    Longitude: float = Field(..., description="Longitude coordinate")


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    rmse: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file house_price_model.joblib not found.")
    return load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(request: PredictionRequest) -> pd.DataFrame:
    data = pd.DataFrame([request.model_dump()])
    return add_features(data)


def predict_batch(model, batch_df: pd.DataFrame) -> pd.DataFrame:
    validated_df = add_features(batch_df.copy())
    raw_predictions = model.predict(validated_df)
    predicted_prices = raw_predictions * 100000

    metrics = load_metrics() or {}
    rmse = float(metrics.get("RMSE", 0.5))
    rmse_dollars = rmse * 100000

    # Approximate 95% prediction interval using 1.96 * RMSE
    interval_width = 1.96 * rmse_dollars

    result_df = batch_df.copy()
    result_df["Predicted Price"] = predicted_prices
    result_df["Confidence Lower"] = np.maximum(predicted_prices - interval_width, 0.0)
    result_df["Confidence Upper"] = predicted_prices + interval_width
    result_df["RMSE"] = rmse_dollars
    return result_df


def make_prediction(model, request: PredictionRequest) -> PredictionResponse:
    data = prepare_data(request)
    raw_prediction = model.predict(data)[0]
    predicted_price = float(raw_prediction * 100000)

    metrics = load_metrics() or {}
    rmse = float(metrics.get("RMSE", 0.5))
    rmse_dollars = rmse * 100000

    # Approximate 95% prediction interval using 1.96 * RMSE
    interval_width = 1.96 * rmse_dollars

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_lower=max(0.0, predicted_price - interval_width),
        confidence_upper=predicted_price + interval_width,
        rmse=rmse_dollars,
    )


@app.get("/", summary="API status")
def root():
    return {"status": "ok", "message": "California House Price API is running."}


@app.get("/health", summary="Health check")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, summary="Predict single house price")
def predict(request: PredictionRequest):
    try:
        model = load_model()
        return make_prediction(model, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, summary="Predict prices for multiple items")
def batch_predict(request: BatchPredictionRequest):
    try:
        model = load_model()
        predictions = [make_prediction(model, item) for item in request.items]
        return BatchPredictionResponse(predictions=predictions)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
