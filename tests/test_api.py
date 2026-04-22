import pandas as pd
from fastapi.testclient import TestClient

import api
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_single():
    payload = {
        "MedInc": 4.0,
        "HouseAge": 20,
        "AveRooms": 5.0,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 2.5,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["confidence_lower"] <= data["predicted_price"] <= data["confidence_upper"]
    assert data["rmse"] > 0


def test_batch_predict_endpoint():
    payload = {
        "items": [
            {
                "MedInc": 4.0,
                "HouseAge": 20,
                "AveRooms": 5.0,
                "AveBedrms": 1.1,
                "Population": 1000,
                "AveOccup": 2.5,
                "Latitude": 34.0,
                "Longitude": -118.0,
            }
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    prediction = data["predictions"][0]
    assert prediction["confidence_lower"] <= prediction["predicted_price"] <= prediction["confidence_upper"]


def test_predict_batch_helper_function():
    model = api.load_model()
    batch_df = pd.DataFrame([
        {
            "MedInc": 4.0,
            "HouseAge": 20,
            "AveRooms": 5.0,
            "AveBedrms": 1.1,
            "Population": 1000,
            "AveOccup": 2.5,
            "Latitude": 34.0,
            "Longitude": -118.0,
        }
    ])
    result = api.predict_batch(model, batch_df)
    assert "Predicted Price" in result.columns
    assert len(result) == 1
    assert result.loc[0, "Predicted Price"] > 0
