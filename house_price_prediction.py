"""End-to-end California house price prediction pipeline.

This script loads the California Housing dataset, performs EDA, creates new features,
trains and tunes a Gradient Boosting Regressor, evaluates performance, and saves a
production-ready model for the Streamlit web application.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from preprocessing import add_features
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_housing_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df["MedHouseVal"] = dataset.target
    
    # Remove outliers that hurt model accuracy
    df = df[df["AveRooms"] < 20]
    df = df[df["AveOccup"] < 10]
    
    return df


def plot_histograms(df: pd.DataFrame, output_path: Path) -> None:
    numeric_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "rooms_per_person",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 10))
    df[numeric_columns].hist(bins=20, figsize=(14, 10), layout=(3, 3))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_correlations(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Matrix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, feature_names: list[str], output_path: Path) -> None:
    # Access the regressor from the pipeline to get feature importances
    importances = model.named_steps['regressor'].feature_importances_
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
    plt.title("Gradient Boosting Feature Importances")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_actual_vs_predicted(y_test, gb_preds, lr_preds, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gradient Boosting
    axes[0].scatter(y_test, gb_preds, alpha=0.3, color="steelblue", s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("Gradient Boosting: Actual vs Predicted")

    # Linear Regression
    axes[1].scatter(y_test, lr_preds, alpha=0.3, color="darkorange", s=10)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")
    axes[1].set_title("Linear Regression: Actual vs Predicted")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(gb_metrics, lr_metrics, output_path: Path) -> None:
    metrics = ["MAE", "RMSE", "R2"]
    gb_values = [gb_metrics[m] for m in metrics]
    lr_values = [lr_metrics[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], gb_values, width, label="Gradient Boosting", color="steelblue")
    bars2 = ax.bar([i + width/2 for i in x], lr_values, width, label="Linear Regression", color="darkorange")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Gradient Boosting vs Linear Regression")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def train_linear_regression(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(df)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    param_grid = {
        "regressor__n_estimators": [150, 250],
        "regressor__max_depth": [3, 4, 5],
        "regressor__learning_rate": [0.05, 0.08, 0.1],
        "regressor__subsample": [0.8, 1.0],
    }
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }


def save_data_ranges(df: pd.DataFrame, output_path: Path) -> None:
    ranges = {}
    for col in df.columns:
        if col != "MedHouseVal":
            ranges[col] = {"min": float(df[col].min()), "max": float(df[col].max())}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=2)


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    df = load_housing_data()
    print("Loaded California Housing dataset")
    print(f"Dataset shape: {df.shape}")

    df = add_features(df)
    plot_histograms(df, Path("eda_histograms.png"))
    plot_feature_correlations(df, Path("correlation_matrix.png"))
    print("Saved EDA charts: eda_histograms.png, correlation_matrix.png")

    save_data_ranges(df, Path("data_ranges.json"))
    print("Saved data ranges to data_ranges.json")

    X, y = prepare_features(df)
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Second split: separate validation set (20% of remaining = 16% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    search = train_model(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV R2: {search.best_score_:.4f}")

    # Validation set performance
    val_metrics = evaluate_model(best_model, X_val, y_val)
    print("\nModel performance on VALIDATION set:")
    for name, value in val_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test set performance
    metrics = evaluate_model(best_model, X_test, y_test)
    print("\nModel performance on TEST set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nBest CV R2 (from GridSearchCV): {search.best_score_:.4f}")

    save_metrics({
        "test": metrics,
        "validation": val_metrics,
        "best_cv_r2": round(search.best_score_, 4)
    }, Path("model_metrics.json"))
    print("Saved metrics to model_metrics.json")

    feature_names = X.columns.tolist()
    plot_feature_importance(best_model, feature_names, Path("feature_importance.png"))
    print("Saved feature importance plot to feature_importance.png")

    # Train Linear Regression model
    print("\nTraining Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print("\nLinear Regression performance on TEST set:")
    for name, value in lr_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save comparison metrics
    save_metrics({
        "gradient_boosting": metrics,
        "linear_regression": lr_metrics,
        "validation": val_metrics,
        "best_cv_r2": round(search.best_score_, 4)
    }, Path("model_metrics.json"))

    # Plot Actual vs Predicted for both models
    gb_preds = best_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)
    plot_actual_vs_predicted(y_test, gb_preds, lr_preds, Path("actual_vs_predicted.png"))
    print("Saved actual vs predicted plot to actual_vs_predicted.png")

    # Plot Model Comparison chart
    plot_model_comparison(metrics, lr_metrics, Path("model_comparison.png"))
    print("Saved model comparison chart to model_comparison.png")

    model_path = Path("house_price_model.joblib")
    dump(best_model, model_path)
    print(f"Saved trained model to {model_path}")

    sample_input = X_test.iloc[0:1]
    sample_price = best_model.predict(sample_input)[0] * 100000
    print("\nSample prediction for first test row:")
    print(sample_input.to_dict(orient="records")[0])
    print(f"Predicted median house value: ${sample_price:,.0f}")


if __name__ == "__main__":
    main()
