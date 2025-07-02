# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model pipeline
model_pipeline = joblib.load('churn_model_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)

    # Convert the JSON data into a pandas DataFrame
    # The keys in the JSON must match the feature names the model was trained on
    input_data = pd.DataFrame([data])

    # Make a prediction
    prediction = model_pipeline.predict(input_data)
    probability = model_pipeline.predict_proba(input_data)

    # Get the churn probability
    churn_prob = probability[0][1] # Probability of class '1' (Churn)

    # Return the response as JSON
    return jsonify({
        'prediction': 'Churn' if prediction[0] == 1 else 'No Churn',
        'churn_probability': f"{churn_prob:.4f}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)