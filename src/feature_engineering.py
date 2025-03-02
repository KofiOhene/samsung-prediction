import pandas as pd

def feature_engineering(df):
    """
    Perform feature engineering on the cleaned dataset:
    - Encode categorical variables
    - Save processed data
    """

    print("Performing Feature Engineering...")

    # Categorical columns that need encoding
    categorical_columns = ["Quarter", "Region", "5G Capability", "Product Model"]

    # One-hot encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Save the feature-engineered dataset
    processed_file_path = "data/processed/featured_data.csv"
    df.to_csv(processed_file_path, index=False)

    print(f"✅ Feature Engineering Complete! Data saved at: {processed_file_path}")

    return df
