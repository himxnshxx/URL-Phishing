from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']
        features = extract_features(url)
        prediction = model.predict([features])

        if prediction[0] == 1:
            result = "URL does not look secure!"
        else:
            result = "URL is safe."

        return render_template('index.html', prediction=result, url=url)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Dummy feature extraction (replace with actual logic)
def extract_features(url):
    return np.random.rand(49)  # Replace with actual feature extraction

if __name__ == "__main__":
    app.run(debug=True)
