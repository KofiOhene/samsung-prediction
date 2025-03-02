import pandas as pd


def feature_engineering(df):
    """
    Perform feature engineering on the cleaned dataset.
    """
    print("\nPerforming Feature Engineering...")

    # Example Feature 1: Create "Revenue per Unit"
    df["Revenue per Unit"] = df["Revenue ($)"] / df["Units Sold"]

    # Example Feature 2: Create "5G Growth Rate"
    df["5G Growth Rate"] = df["5G Subscribers (millions)"].pct_change().fillna(0)

    # Example Feature 3: Convert "Quarter" to numerical
    quarter_mapping = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df["Quarter_Num"] = df["Quarter"].map(quarter_mapping)

    # Save processed dataset
    processed_file_path = "data/processed/featured_data.csv"
    df.to_csv(processed_file_path, index=False)
    print(f"\nFeature Engineering Complete! Data saved at: {processed_file_path}")

    return df
