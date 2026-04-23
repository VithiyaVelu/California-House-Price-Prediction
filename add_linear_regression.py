content = open("house_price_prediction.py").read()

# 1. Add LinearRegression to imports
old_imports = "from sklearn.ensemble import GradientBoostingRegressor"
new_imports = """from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression"""

# 2. Add comparison plot function after plot_feature_importance function
old_func = "def prepare_features(df: pd.DataFrame)"
new_func = """def plot_actual_vs_predicted(y_test, gb_preds, lr_preds, output_path: Path) -> None:
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


def prepare_features(df: pd.DataFrame)"""

# 3. Add Linear Regression training + plots in main() after saving metrics
old_main = """    model_path = Path("house_price_model.joblib")
    dump(best_model, model_path)
    print(f"Saved trained model to {model_path}")"""

new_main = """    # Train Linear Regression model
    print("\\nTraining Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print("\\nLinear Regression performance on TEST set:")
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
    print(f"Saved trained model to {model_path}")"""

result = content
result = result.replace(old_imports, new_imports)
result = result.replace(old_func, new_func)
result = result.replace(old_main, new_main)

open("house_price_prediction.py", "w").write(result)
print("Done! Linear Regression + comparison graphs added successfully!")
