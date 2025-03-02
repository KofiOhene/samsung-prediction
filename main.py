from src.data_cleaning import load_and_clean_data

# Define file path
file_path = "data/raw/Expanded_Dataset.csv"

# Load and clean data
df_cleaned = load_and_clean_data(file_path)

# Display summary of cleaned data
print("\nFinal Cleaned Data Overview:")
print(df_cleaned.describe(include='all'))
