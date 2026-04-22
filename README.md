# House Price Prediction

This repository contains a complete end-to-end machine learning pipeline and a Streamlit web application for California housing prices.

## What it does

- Loads and explores the California Housing dataset
- Creates engineered features like `rooms_per_person`
- Trains and tunes a `GradientBoostingRegressor`
- Evaluates with MAE, RMSE, and R²
- Saves the trained model to `house_price_model.joblib`
- Provides an interactive Streamlit app in `app.py`

## Files

- `house_price_prediction.py` - full training pipeline, EDA, feature engineering, model tuning, and persistence
- `app.py` - Streamlit web app for interactive predictions
- `requirements.txt` - Python dependencies
- `correlation_matrix.png` - generated after running the training script
- `eda_histograms.png` - generated after running the training script
- `feature_importance.png` - generated after running the training script
- `model_metrics.json` - generated after running the training script
- `house_price_model.joblib` - saved model after training

## Usage

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Train the model and generate artifacts

```powershell
python house_price_prediction.py
```

4. Launch the Streamlit app

```powershell
streamlit run app.py
```

5. Launch the API backend

```powershell
uvicorn api:app --reload --port 8000
```

## Automated Tests

Install test dependencies and run tests with pytest:

```powershell
pip install -r requirements.txt
pytest
```

## Docker Deployment

Build the Docker image:

```powershell
docker build -t house-price-predictor .
```

Run the Streamlit app container:

```powershell
docker run --rm -p 8501:8501 house-price-predictor
```

Run the API container:

```powershell
docker run --rm -p 8000:8000 house-price-predictor uvicorn api:app --host 0.0.0.0 --port 8000
```

Or launch both services together with Docker Compose:

```powershell
docker compose up --build
```

## API Usage

- Single prediction:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{
  \"MedInc\": 4.0,
  \"HouseAge\": 20,
  \"AveRooms\": 5.0,
  \"AveBedrms\": 1.1,
  \"Population\": 1000,
  \"AveOccup\": 2.5,
  \"Latitude\": 34.0,
  \"Longitude\": -118.0
}"
```

- Batch prediction:

```powershell
curl -X POST "http://127.0.0.1:8000/batch_predict" -H "Content-Type: application/json" -d "{
  \"items\": [
    {
      \"MedInc\": 4.0,
      \"HouseAge\": 20,
      \"AveRooms\": 5.0,
      \"AveBedrms\": 1.1,
      \"Population\": 1000,
      \"AveOccup\": 2.5,
      \"Latitude\": 34.0,
      \"Longitude\": -118.0
    }
  ]
}"
```

## Deployment

To expose the locally running Streamlit app via a public URL, install `localtunnel` with Node.js and run:

```powershell
npm install -g localtunnel
lt --port 8501
```

Then open the URL returned by `localtunnel`.

## Requirements

- Python 3.9+
- streamlit
- scikit-learn
- pandas
- seaborn
- matplotlib
- joblib
