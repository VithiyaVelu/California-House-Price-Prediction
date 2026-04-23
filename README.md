# 🏠 California House Price Prediction App

A comprehensive machine learning application for predicting California housing prices with an interactive Streamlit web interface.

## ✨ Features

### 🤖 Machine Learning Pipeline
- **Data Exploration**: Comprehensive EDA with visualizations
- **Feature Engineering**: Advanced feature creation (rooms_per_person, bedrooms_per_room, etc.)
- **Model Training**: Gradient Boosting Regressor with hyperparameter tuning
- **Model Evaluation**: MAE, RMSE, R² metrics with cross-validation
- **Model Persistence**: Saved model for production use

### 🌐 Interactive Web App
- **Single Property Prediction**: Real-time price estimation with sliders
- **What-if Analysis**: Interactive feature impact visualization
- **Batch CSV Prediction**: Process multiple properties simultaneously
- **Model Performance Analysis**: Actual vs predicted price comparison with metrics
- **PDF Report Generation**: Professional downloadable reports
- **CSV Template Download**: Ready-to-use data templates

### 📊 Advanced Analytics
- **Comparison Metrics**: MAE, RMSE, R², accuracy percentages
- **Visual Analytics**: Scatter plots and error distribution histograms
- **Confidence Intervals**: Prediction ranges with model uncertainty
- **Feature Importance**: Understanding model decision factors

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd house_priceprediction
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python house_price_prediction.py
```

5. **Launch the web app**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
house_priceprediction/
├── app.py                          # Main Streamlit application
├── house_price_prediction.py       # ML training pipeline
├── preprocessing.py                # Feature engineering functions
├── pdf_generator.py               # PDF report generation
├── api.py                         # FastAPI backend (optional)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                    # Git ignore rules
├── batch_prediction_template.csv # CSV template for batch predictions
├── sample_with_actual_prices.csv # Sample data with actual prices
├── house_price_model.joblib      # Trained ML model
├── model_metrics.json            # Model performance metrics
├── data_ranges.json              # Feature value ranges
├── tests/                        # Unit tests
│   ├── test_api.py
│   └── test_preprocessing.py
├── __pycache__/                  # Python cache (ignored)
└── *.png                         # Generated visualizations
```

## 🎯 Usage Guide

### Single Property Prediction
1. Adjust sliders for property characteristics
2. View real-time price prediction
3. Use "What-if Analysis" to see feature impacts
4. Download PDF report

### Batch Prediction
1. Download CSV template or use sample data
2. Fill in property details (add `ActualPrice` column for comparison)
3. Upload CSV file
4. View predictions and performance analysis
5. Download results as CSV

### Model Validation
- Upload CSV with actual sale prices
- Get comprehensive performance metrics
- View scatter plots and error distributions
- Assess model accuracy and reliability

## 🔧 API Endpoints (Optional)

The project includes a FastAPI backend for programmatic access:

```python
# Start API server
uvicorn api:app --reload

# Example API call
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}'
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🐳 Docker Deployment

Build and run with Docker:
```bash
docker build -t house-price-prediction .
docker run -p 8501:8501 house-price-prediction
```

## 📈 Model Performance


Current model metrics (after training):

| Metric | Value | Meaning |
|--------|-------|---------|
| **R² Score** | 0.8107 | Model explains 81% of price variance |
| **MAE** | $33,558 | Average prediction error |
| **RMSE** | $49,807 | Root mean squared error |

> Model: Gradient Boosting Regressor (GridSearchCV tuned)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- California Housing Dataset from scikit-learn
- Streamlit for the web framework
- ReportLab for PDF generation
- scikit-learn for machine learning tools

---

**Built with ❤️ for real estate professionals and data enthusiasts**
