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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def load_housing_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df["MedHouseVal"] = dataset.target
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


def plot_feature_importance(model: GradientBoostingRegressor, feature_names: list[str], output_path: Path) -> None:
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
    plt.title("Gradient Boosting Feature Importances")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(df)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    param_grid = {
        "n_estimators": [150, 250],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.08, 0.1],
        "subsample": [0.8, 1.0],
    }
    model = GradientBoostingRegressor(random_state=42)
    search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(model: GradientBoostingRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    search = train_model(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV R2: {search.best_score_:.4f}")

    metrics = evaluate_model(best_model, X_test, y_test)
    print("\nModel performance on test set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    save_metrics(metrics, Path("model_metrics.json"))
    print("Saved metrics to model_metrics.json")

    feature_names = X.columns.tolist()
    plot_feature_importance(best_model, feature_names, Path("feature_importance.png"))
    print("Saved feature importance plot to feature_importance.png")

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
