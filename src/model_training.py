import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_revenue_model(file_path):
    """
    Train a regression model to predict revenue.
    """

    print("\n Loading Data for Training...")
    df = pd.read_csv(file_path)

    # Define Features and Target Variable
    features = df.drop(columns=["Revenue ($)", "Product Model"])  # Exclude target & identifier columns
    target = df["Revenue ($)"]

    # Convert categorical variables using One-Hot Encoding
    features = pd.get_dummies(features, columns=["5G Capability", "Quarter", "Region"], drop_first=True)

    # Train-Test Split
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    print(f"\n🔹 Training Set: {features_train.shape[0]} samples")
    print(f"🔹 Test Set: {features_test.shape[0]} samples")

    # Train a Linear Regression Model
    print("\n🚀 Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(features_train, target_train)

    # Predictions
    target_pred = model.predict(features_test)

    # Evaluate Model Performance
    mae = mean_absolute_error(target_test, target_pred)
    mse = mean_squared_error(target_test, target_pred)
    r2 = r2_score(target_test, target_pred)

    print("\nModel Training Complete!")
    print(f"\n Model Evaluation:")
    print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
    print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
    print(f"🔹 R² Score: {r2:.4f}")

    return model, features_test, target_test, target_pred
