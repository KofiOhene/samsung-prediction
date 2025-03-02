import pandas as pd


def load_and_clean_data(file_path):
    """
    Load dataset, identify and clean invalid negative values, cap percentage columns,
    ensure categorical consistency, and remove duplicates.
    """
    print("Loading Dataset...")
    df = pd.read_csv(file_path)

    # Display initial summary
    print("\nInitial Data Overview:")
    print(df.describe(include='all'))

    # Step 1: Fix Negative Values ###
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

    # Step 2: Cap Percentage Values at 100% ###
    percentage_columns = ["Regional 5G Coverage (%)", "Market Share (%)"]

    for col in percentage_columns:
        num_exceeding = (df[col] > 100).sum()
        print(f"\n️ {num_exceeding} values exceed 100% in '{col}'.")

        # Cap values at 100%
        df[col] = df[col].apply(lambda x: min(x, 100))
        print(f"Capped '{col}' values at 100%.")

    # Step 3: Validate Categorical Data ###
    categorical_columns = ["5G Capability", "Quarter", "Region"]

    for col in categorical_columns:
        unique_values = df[col].unique()
        print(f"\n🔍 Unique values in '{col}': {unique_values}")

    # Step 4: Remove Duplicates ###
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_rows = df.shape[0]

    print(f"\nRemoved {initial_rows - final_rows} duplicate rows.")

    # Step 5: Save Cleaned Data ###
    cleaned_file_path = "data/processed/cleaned_data.csv"
    df.to_csv(cleaned_file_path, index=False)
    print(f"\nCleaned data saved at: {cleaned_file_path}")

    return df
