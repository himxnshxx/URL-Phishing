import pandas as pd

def preprocess_data(input_path, output_path):
    # Load the dataset with no headers
    df = pd.read_csv(input_path, header=None)
    print("Data Loaded. Shape:", df.shape)

    # Assume the last column is the label (0 = Legit, 1 = Phishing)
    df.columns = [f'feature_{i}' for i in range(df.shape[1] - 1)] + ['label']

    print("Columns Renamed. First 5 Rows:")
    print(df.head())

    # Save the preprocessed data
    df.to_csv(output_path, index=False)
    print("Preprocessed data saved to:", output_path)

# File paths
input_path = 'dataset/Phishing_Legitimate_full.csv'
output_path = 'dataset/preprocessed_data.csv'

preprocess_data(input_path, output_path)
