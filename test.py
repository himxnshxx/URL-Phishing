import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed test data
df = pd.read_csv('dataset/preprocessed_data.csv')

# Ensure label exists
if 'label' not in df.columns:
    raise ValueError("Label column not found in the dataset!")

# Split into features and labels
X = df.drop('label', axis=1)
y = df['label']

# Ensure data is clean
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Load the trained model
with open('model/phishing_model.pkl', 'rb') as file:
    model = pickle.load(file)
print("Model Loaded Successfully!")

# Check feature consistency
if X.shape[1] != model.n_features_in_:
    raise ValueError(f"Feature Mismatch! Model expects {model.n_features_in_} features, but found {X.shape[1]}")

# Predict
y_pred = model.predict(X)

# Evaluation
print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
