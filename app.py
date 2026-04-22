import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from pathlib import Path
from joblib import load
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime

from preprocessing import add_features
from pdf_generator import generate_pdf_report

MODEL_PATH = Path(__file__).with_name("house_price_model.joblib")
METRICS_PATH = Path(__file__).with_name("model_metrics.json")
RANGES_PATH = Path(__file__).with_name("data_ranges.json")


@st.cache_resource
def load_model() -> Optional[object]:
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Please run `python house_price_prediction.py` to create `house_price_model.joblib`."
        )
        return None

    try:
        return load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


def load_metrics() -> Optional[dict]:
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_data_ranges() -> Optional[dict]:
    if not RANGES_PATH.exists():
        return None
    try:
        with open(RANGES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def check_input_ranges(input_data: dict, ranges: dict) -> list[str]:
    warnings = []
    for key, value in input_data.items():
        if key in ranges:
            min_val = ranges[key]["min"]
            max_val = ranges[key]["max"]
            if value < min_val or value > max_val:
                warnings.append(f"{key}: {value:.2f} is outside training range [{min_val:.2f}, {max_val:.2f}]")
    return warnings



def predict_price(model: object, inputs: dict) -> float:
    df = pd.DataFrame([inputs])
    df = add_features(df)
    prediction = model.predict(df)[0]
    return prediction * 100000


def generate_whatif_data(model: object, base_inputs: dict, feature: str, values: list[float]) -> pd.DataFrame:
    records = []
    for value in values:
        scenario = base_inputs.copy()
        scenario[feature] = value
        df = pd.DataFrame([scenario])
        df = add_features(df)
        prediction = model.predict(df)[0] * 100000
        records.append({feature: value, "Predicted Price": prediction})
    return pd.DataFrame(records)


def predict_batch(model: object, batch_df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[dict]]:
    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    missing = [col for col in required_columns if col not in batch_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = batch_df.copy()
    try:
        data[required_columns] = data[required_columns].astype(float)
    except Exception as exc:
        raise ValueError("Batch CSV contains non-numeric values in required columns.") from exc

    features = add_features(data[required_columns])
    predictions = model.predict(features) * 100000
    data["Predicted Price"] = predictions
    
    # Check for actual prices
    actual_price_cols = ["ActualPrice", "MedHouseVal", "Actual_Price", "SalePrice", "Price"]
    actual_col = None
    for col in actual_price_cols:
        if col in data.columns:
            actual_col = col
            break
    
    comparison_metrics = None
    if actual_col:
        try:
            actual_prices = data[actual_col].astype(float)
            data["Actual Price"] = actual_prices
            data["Price Difference"] = data["Predicted Price"] - actual_prices
            data["Absolute Error"] = abs(data["Price Difference"])
            data["Percentage Error"] = (data["Absolute Error"] / actual_prices) * 100
            
            # Calculate comparison metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(actual_prices, predictions)
            mse = mean_squared_error(actual_prices, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_prices, predictions)
            
            comparison_metrics = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Mean Actual Price": actual_prices.mean(),
                "Mean Predicted Price": predictions.mean(),
                "Accuracy within 10%": (data["Percentage Error"] <= 10).mean() * 100,
                "Accuracy within 20%": (data["Percentage Error"] <= 20).mean() * 100,
            }
        except Exception as e:
            st.warning(f"Could not process actual prices from column '{actual_col}': {e}")
    
    return data, comparison_metrics


def load_map_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df["MedHouseVal"] = dataset.target * 100000
    df = df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    return df[["lat", "lon", "MedHouseVal"]]


def generate_pdf_report(report_text: str, filename: str = "house_price_prediction_report.pdf") -> bytes:
    """Generate a PDF report from the text report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=6,
        spaceBefore=6
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=12
    )
    
    elements = []
    
    # Add title
    elements.append(Paragraph("🏠 California House Price Prediction Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Parse the report text and add to PDF
    lines = report_text.split('\n')
    for line in lines:
        if line.startswith('==='):
            continue
        elif line.startswith('- '):
            # Bullet point
            elements.append(Paragraph(f"• {line[2:]}", normal_style))
        elif any(line.startswith(header) for header in ['Prediction Details:', 'Input Features:', 'Engineered Features:', 'What-if Analysis', 'Generated on:']):
            # Section heading
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(line, heading_style))
        elif line.strip():
            # Regular text
            elements.append(Paragraph(line, normal_style))
        else:
            # Empty line
            elements.append(Spacer(1, 0.05*inch))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def main() -> None:
    st.set_page_config(
        page_title="California House Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3561;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
    }
    .warning-box {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        border: 2px solid #ff9a56;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2d3561;
        font-weight: 600;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .confidence-range {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .download-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    .slider-label {
        font-weight: 600;
        color: #2d3561;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🏠 California House Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
    🔮 AI-Powered Real-Time Estate Valuation | Instant Price Predictions
    </p>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🎛️ Input Features")
        st.markdown("Adjust the sliders to input neighborhood characteristics:")

        col1, col2 = st.columns(2)
        with col1:
            median_income = st.slider("Median Income (tens of thousands)", 0.0, 20.0, 4.0, 0.1, help="Median household income in the area")
            house_age = st.slider("House Age (years)", 1, 80, 20, help="Average age of houses in the neighborhood")
            avg_rooms = st.slider("Average Rooms", 1.0, 20.0, 5.0, 0.1, help="Average number of rooms per household")
        with col2:
            avg_bedrooms = st.slider("Average Bedrooms", 0.5, 10.0, 1.1, 0.1, help="Average number of bedrooms per household")
            population = st.slider("Population", 100, 5000, 1000, help="Total population in the neighborhood")
            avg_occupancy = st.slider("Average Occupancy", 1.0, 8.0, 2.5, 0.1, help="Average household size")

        st.subheader("📍 Location")
        latitude = st.slider("Latitude", 32.0, 42.0, 34.0, 0.01, help="Geographic latitude of the neighborhood")
        longitude = st.slider("Longitude", -125.0, -114.0, -118.0, 0.01, help="Geographic longitude of the neighborhood")

    input_data = {
        "MedInc": median_income,
        "HouseAge": house_age,
        "AveRooms": avg_rooms,
        "AveBedrms": avg_bedrooms,
        "Population": population,
        "AveOccup": avg_occupancy,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    # Check for input range warnings
    ranges = load_data_ranges()
    if ranges:
        warnings = check_input_ranges(input_data, ranges)
        if warnings:
            st.markdown('<div class="warning-box">⚠️ <strong>Input Range Warning:</strong> Some values are outside training data ranges:</div>', unsafe_allow_html=True)
            for warning in warnings:
                st.write(f"• {warning}")

    model = load_model()
    if model is None:
        return

    estimated_price = predict_price(model, input_data)

    # Load metrics for confidence range
    metrics = load_metrics()
    rmse = metrics.get("RMSE", 0.5) if metrics else 0.5  # fallback RMSE
    rmse_dollars = rmse * 100000  # convert to dollars
    lower_bound = max(0, estimated_price - rmse_dollars)
    upper_bound = estimated_price + rmse_dollars

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Analysis", "ℹ️ About"])

    with tab1:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Real-time prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <div style="text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">💰 Estimated Median House Value</div>
                <div class="prediction-value">${estimated_price:,.0f}</div>
                <div class="confidence-range">📊 Confidence Range: <strong>${lower_bound:,.0f}</strong> → <strong>${upper_bound:,.0f}</strong></div>
                <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">±${rmse_dollars:,.0f} (Model RMSE)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Status", "Ready", "Real-time")
        with col2:
            st.metric("🎲 Model", "Gradient Boosting", "R² 0.81")
        with col3:
            st.metric("⚡ Features", "13 Engineered", "Active")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Input Values")
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df.T.rename(columns={0: "Value"}), width='stretch')

        with col2:
            st.subheader("🔧 Engineered Features")
            engineered_df = add_features(pd.DataFrame([input_data]))
            engineered_features = engineered_df.drop(columns=list(input_data.keys()))
            st.dataframe(engineered_features.T.rename(columns={0: "Value"}), width='stretch')

        st.markdown("---")
        st.subheader("💡 What-if Analysis")
        st.write("*Adjust one feature to see instant impact on predicted price*")
        
        whatif_feature = st.selectbox(
            "Choose a feature to vary",
            options=[
                "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
                "Latitude", "Longitude"
            ],
            index=0,
            help="Select one feature to see how changing it affects the predicted price."
        )

        if ranges and whatif_feature in ranges:
            feature_min = ranges[whatif_feature]["min"]
            feature_max = ranges[whatif_feature]["max"]
        else:
            feature_min, feature_max = float(input_data[whatif_feature]) * 0.5, float(input_data[whatif_feature]) * 1.5

        values = np.linspace(feature_min, feature_max, 15)
        whatif_df = generate_whatif_data(model, input_data, whatif_feature, values)
        whatif_df = whatif_df.set_index(whatif_feature)

        st.line_chart(whatif_df, width='stretch')

        current_prediction = predict_price(model, input_data)
        best_row = whatif_df.loc[whatif_df["Predicted Price"].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📍 Current: {whatif_feature} = {input_data[whatif_feature]:,.2f}")
            st.write(f"Price: **${current_prediction:,.0f}**")
        with col2:
            st.success(f"🎯 Best in range: {whatif_feature} = {best_row.name:,.2f}")
            st.write(f"Price: **${best_row['Predicted Price']:,.0f}**")

        st.markdown("---")
        st.subheader("📁 Batch CSV Prediction")
        st.write("Upload a CSV file with the original input columns to generate batch predictions.")
        st.write("Required columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.")
        
        # Template download button
        with open("batch_prediction_template.csv", "rb") as f:
            template_data = f.read()
        st.download_button(
            label="📋 Download CSV Template",
            data=template_data,
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            help="Download a template CSV file with required columns and sample data"
        )
        
        st.markdown("---")

        uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], help="CSV must include the original feature columns. Optional: Add 'ActualPrice' column for comparison analysis.")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                batch_result, comparison_metrics = predict_batch(model, batch_df)
                st.success(f"Batch prediction completed for {len(batch_result)} rows.")
                
                # Display results
                st.dataframe(batch_result, width='stretch')
                
                # Show comparison metrics if available
                if comparison_metrics:
                    st.markdown("---")
                    st.subheader("📊 Model Performance Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"${comparison_metrics['MAE']:,.0f}", "Mean Absolute Error")
                    with col2:
                        st.metric("RMSE", f"${comparison_metrics['RMSE']:,.0f}", "Root Mean Squared Error")
                    with col3:
                        st.metric("R² Score", f"{comparison_metrics['R2']:.3f}", "Coefficient of Determination")
                    with col4:
                        st.metric("Accuracy ±10%", f"{comparison_metrics['Accuracy within 10%']:.1f}%", "Within 10% of actual")
                    
                    # Additional metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"Mean Actual Price: ${comparison_metrics['Mean Actual Price']:,.0f}")
                        st.info(f"Accuracy ±20%: {comparison_metrics['Accuracy within 20%']:.1f}%")
                    with col2:
                        st.info(f"Mean Predicted Price: ${comparison_metrics['Mean Predicted Price']:,.0f}")
                    
                    # Scatter plot comparison
                    st.markdown("### Actual vs Predicted Prices")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(batch_result["Actual Price"], batch_result["Predicted Price"], alpha=0.6, color='#667eea')
                    ax.plot([batch_result["Actual Price"].min(), batch_result["Actual Price"].max()], 
                           [batch_result["Actual Price"].min(), batch_result["Actual Price"].max()], 
                           'r--', label='Perfect Prediction')
                    ax.set_xlabel('Actual Price ($)')
                    ax.set_ylabel('Predicted Price ($)')
                    ax.set_title('Actual vs Predicted House Prices')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Error distribution
                    st.markdown("### Prediction Error Distribution")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.hist(batch_result["Percentage Error"], bins=30, alpha=0.7, color='#764ba2', edgecolor='black')
                    ax2.set_xlabel('Percentage Error (%)')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Distribution of Prediction Errors')
                    ax2.axvline(x=10, color='red', linestyle='--', label='±10% threshold')
                    ax2.axvline(x=-10, color='red', linestyle='--')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                
                csv_data = batch_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Predictions",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

        # Download report
        report = f"""
California House Price Prediction Report
=======================================

Prediction Details:
- Estimated Median House Value: ${estimated_price:,.0f}
- Confidence Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}
- Model RMSE: ${rmse_dollars:,.0f}

Input Features:
{chr(10).join(f"- {k}: {v:,.2f}" for k, v in input_data.items())}

Engineered Features:
{chr(10).join(f"- {k}: {v:,.4f}" for k, v in engineered_features.iloc[0].to_dict().items())}

What-if Analysis (Feature: {whatif_feature}):
- Current Value: {input_data[whatif_feature]:,.2f}
- Best in Range: {best_row.name:,.2f} (${best_row['Predicted Price']:,.0f})

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()

        st.markdown("---")
        # Generate PDF report
        pdf_data = generate_pdf_report(report)
        st.download_button(
            label="📥 Download Prediction Report (PDF)",
            data=pdf_data,
            file_name="house_price_prediction_report.pdf",
            mime="application/pdf",
            help="Download a detailed PDF report of your prediction"
        )

    with tab2:
        st.markdown('<h2 class="sub-header">📊 Model Analysis & Performance</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("📏 Performance Metrics")
            metrics = load_metrics()
            if metrics:
                st.metric("MAE", f"${metrics['MAE']*100000:,.0f}", "Mean Absolute Error")
                st.metric("RMSE", f"${metrics['RMSE']*100000:,.0f}", "Root Mean Squared Error")
                st.metric("R² Score", f"{metrics['R2']:.2%}", "Coefficient of Determination")
            else:
                st.write("Metrics not available")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🎯 Top Features")
            if hasattr(model, "feature_importances_"):
                feature_names = [
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
                    "Latitude", "Longitude", "rooms_per_person", "bedrooms_per_room",
                    "population_per_household", "population_per_room", "medinc_houseage_interaction"
                ]
                importances = model.feature_importances_
                top_features = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(5)
                st.bar_chart(top_features.set_index("Feature"), width='stretch')
            else:
                st.write("Not available")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("📈 Model Specs")
            st.write("• **Model:** Gradient Boosting")
            st.write("• **Dataset:** California Housing")
            st.write("• **Records:** 20,640")
            st.write("• **Features:** 13 Engineered")
            st.write("• **Train/Test:** 80/20 split")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('---')
        st.subheader('🗺️ California Housing Map')
        st.write('Explore real housing locations from the California Housing dataset, with predicted price highlights.')

        map_data = load_map_data()
        current_point = pd.DataFrame([
            {"lat": latitude, "lon": longitude, "MedHouseVal": estimated_price, "type": "Current Input"}
        ])
        map_df = pd.concat([map_data, current_point], ignore_index=True)

        dataset_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["lon", "lat"],
            get_color="[0, 112, 255, 120]",
            get_radius="MedHouseVal / 70",
            pickable=True,
            auto_highlight=True,
        )

        current_layer = pdk.Layer(
            "ScatterplotLayer",
            data=current_point,
            get_position=["lon", "lat"],
            get_color="[255, 80, 80, 200]",
            get_radius=3000,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=map_data["lat"].mean(),
            longitude=map_data["lon"].mean(),
            zoom=5,
            pitch=30,
        )

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[dataset_layer, current_layer],
            tooltip={"text": "Price: ${MedHouseVal}"},
        )

        st.pydeck_chart(deck)

    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 Features
            - **Real-time Predictions:** Instant price estimates as you adjust inputs
            - **What-if Analysis:** Explore feature impact with interactive charts
            - **Confidence Ranges:** See prediction uncertainty (±RMSE)
            - **Feature Importance:** Understand which factors drive prices
            - **Downloadable Reports:** Export predictions with full details
            
            ### 🛠️ Technical Stack
            - **Framework:** Streamlit
            - **ML Model:** scikit-learn Gradient Boosting
            - **Data Processing:** pandas, numpy
            - **Visualization:** Streamlit charts
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Model Details
            - **Training Data:** California Housing Dataset (20,640 records)
            - **Features:** 8 original + 5 engineered features
            - **Model Type:** GradientBoostingRegressor
            - **Test R²:** 0.81 (good predictive power)
            - **Validation:** 5-fold cross-validation
            
            ### 💡 Use Cases
            - Real estate market analysis
            - Property valuation
            - Investment decisions
            - Market trend exploration
            """)

        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;">
        <h3 style="color: white; margin: 0;">🌟 Production-Ready ML Application</h3>
        <p style="margin: 1rem 0 0 0;">Built with enterprise-grade error handling, caching, and user-friendly interface</p>
        </div>
        """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
