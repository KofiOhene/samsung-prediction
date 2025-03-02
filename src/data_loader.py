import pandas as pd

def load_data(file_path):
    """
    Load dataset and display initial statistics.
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    # Show dataset info
    print("\nDataset Overview:")
    print(df.info())

    # Show first few rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    return df

if __name__ == "__main__":
    file_path = "data/raw/Expanded_Dataset.csv"  # Ensure your dataset is in the 'data' folder
    df = load_data(file_path)
