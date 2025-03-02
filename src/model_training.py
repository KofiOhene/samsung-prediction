import pandas as pd
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def train_and_visualize_models(featured_data_path):
    print("Starting Model Training & Visualization...")

    # Load dataset
    df = pd.read_csv(featured_data_path)

    # Define features and target
    feature_columns = [col for col in df.columns if col not in ['Revenue ($)']]
    target_column = 'Revenue ($)'

    # Split data
    train_features, test_features, train_target, test_target = train_test_split(
        df[feature_columns], df[target_column], test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    model_results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...\n")
        model.fit(train_features, train_target)

        # Predictions
        predictions = model.predict(test_features)

        # Evaluation Metrics
        mae = mean_absolute_error(test_target, predictions)
        mse = mean_squared_error(test_target, predictions)
        r2 = r2_score(test_target, predictions)

        print(f"{model_name} Training Complete!\n")
        print(f"{model_name} Performance:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.4f}\n")

        # Save the model
        model_filename = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Model saved successfully at: {model_filename}")

        # Store results
        model_results.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "R² Score": r2
        })

        # Feature Importance (Only for Tree-Based Models)
        if model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            print(f"🔹 Top Features impacting Revenue ({model_name}):")
            print(feature_importance_df.head(10))

            # Feature Importance Plot
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='blue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.savefig(f"models/feature_importance_{model_name.replace(' ', '_').lower()}.png")
            plt.show()

        # Residual Plot
        residuals = test_target - predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Revenue")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot - {model_name}")
        plt.savefig(f"models/residual_plot_{model_name.replace(' ', '_').lower()}.png")
        plt.show()

        # Error Distribution Plot
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=30, color='blue', alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Error Distribution - {model_name}")
        plt.savefig(f"models/error_distribution_{model_name.replace(' ', '_').lower()}.png")
        plt.show()

    # Save results as CSV
    results_df = pd.DataFrame(model_results)
    results_df.to_csv("models/model_comparison.csv", index=False)
    print("\nModel performance comparison saved at: models/model_comparison.csv")

    return results_df
