import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load and validate data
df = pd.read_csv('dataset/preprocessed_data.csv')
print("Data Loaded. Shape:", df.shape)

X = df.drop('label', axis=1)
y = df['label']

# Ensure numeric data
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print("Unique labels in y:", y.unique())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=500, solver='liblinear')
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the Model
with open('model/phishing_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully!")
