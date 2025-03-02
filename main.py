import pandas as pd
from src.data_cleaning import load_and_clean_data
from src.feature_engineering import feature_engineering
from src.model_training import train_revenue_model

# Define file path for processed dataset
processed_data_path = "data/processed/featured_data.csv"

# Train Revenue Prediction Model
print("\nStarting Revenue Prediction Model Training...")
model, features_test, target_test, target_pred = train_revenue_model(processed_data_path)
