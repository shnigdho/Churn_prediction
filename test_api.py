# test_api.py
import requests
import json

# URL of our local Flask app
URL = 'http://127.0.0.1:5000/predict'

# Example customer data (must match the model's expected features)
# This is one row from our original dataset
customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

# Send a POST request with the data
response = requests.post(URL, json=customer_data)

# Print the result
print("API Response:")
print(response.json())