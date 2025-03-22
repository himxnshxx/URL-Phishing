import pandas as pd

# Try to read without headers
df = pd.read_csv('dataset/Phishing_Legitimate_full.csv', header=None)
print("First 5 rows of the dataset (no headers assumed):")
print(df.head())
print("\nNumber of columns:", len(df.columns))
