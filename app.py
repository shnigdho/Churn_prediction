from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the tuned model
model_pipeline = joblib.load('churn_model_tuned_xgb.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    form_data = request.form.to_dict()
    
    # Convert numerical fields from string to number
    form_data['tenure'] = int(form_data['tenure'])
    form_data['MonthlyCharges'] = float(form_data['MonthlyCharges'])
    # You MUST convert all numeric/float features here
    form_data['SeniorCitizen'] = int(form_data['SeniorCitizen'])
    form_data['TotalCharges'] = float(form_data['TotalCharges'])


    # Convert to DataFrame
    input_data = pd.DataFrame([form_data])
    
    # Make prediction
    prediction = model_pipeline.predict(input_data)[0]
    probability = model_pipeline.predict_proba(input_data)[0][1]

    # Prepare response
    pred_text = "Prediction: Customer will Churn" if prediction == 1 else "Prediction: Customer will Not Churn"
    prob_text = f"Probability of Churn: {probability:.2%}"

    return render_template('index.html', prediction_text=pred_text, probability_text=prob_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)