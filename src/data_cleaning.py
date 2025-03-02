import pandas as pd
import os


def load_and_clean_data(file_path):
    """
    Load dataset, identify and clean invalid negative values, cap percentage columns,
    ensure categorical consistency, and retain all data without removing duplicates.
    """
    print("Loading Dataset...")
    df = pd.read_csv(file_path)

    # Display initial summary
    print("\nInitial Data Overview:")
    print(df.describe(include='all'))

    # Step 1: Identify and Verify Potential Duplicates ###
    potential_duplicates = df[df.duplicated(subset=["Year", "Quarter", "Region"], keep=False)]

    # Count unique product models per duplicate group
    product_variation = potential_duplicates.groupby(["Year", "Quarter", "Region"])[
        "Product Model"].nunique().reset_index()

    # Find where multiple products exist per quarter-region combination
    multi_product_cases = product_variation[product_variation["Product Model"] > 1]

    print(f"Found {len(multi_product_cases)} cases where multiple products exist in the same quarter and region.")

    # Display sample
    print("Sample cases with multiple product models per quarter/region:")
    print(multi_product_cases.head(10))

    # Step 2: Fix Negative Values ###
    columns_with_negatives = ["5G Subscribers (millions)", "Market Share (%)"]

    for col in columns_with_negatives:
        num_negatives = (df[col] < 0).sum()
        print(f"\n {num_negatives} negative values found in '{col}'.")

        # Replace negative values with NaN
        df[col] = df[col].apply(lambda x: x if x >= 0 else None)

        # Impute missing values with median
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Imputed missing values in '{col}' using median: {median_value}")

    # Step 3: Cap Percentage Values at 100% ###
    percentage_columns = ["Regional 5G Coverage (%)", "Market Share (%)"]

    for col in percentage_columns:
        num_exceeding = (df[col] > 100).sum()
        print(f"\n {num_exceeding} values exceed 100% in '{col}'.")

        # Cap values at 100%
        df[col] = df[col].apply(lambda x: min(x, 100))
        print(f"Capped '{col}' values at 100%.")

    # Step 4: Validate Categorical Data ###
    categorical_columns = ["5G Capability", "Quarter", "Region"]

    for col in categorical_columns:
        unique_values = df[col].unique()
        print(f"\nUnique values in '{col}': {unique_values}")

    # Step 5: Retain All Data (Skip Duplicate Removal) ###
    print("\nDuplicate removal skipped: Retaining all data.")

    # Step 6: Save Cleaned Data ###
    cleaned_file_path = "data/processed/cleaned_data.csv"
    os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
    df.to_csv(cleaned_file_path, index=False)
    print(f"\nCleaned data saved at: {cleaned_file_path}")

    return df
