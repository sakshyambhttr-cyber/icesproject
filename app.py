#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Customer Churn Prediction Flask App
Models: Random Forest for churn prediction
Features: Age, Tenure, Monthly Charges, Gender
"""

from flask import Flask, request, render_template
import pickle
import numpy as np

print("\n" + "="*60)
print("STARTING CUSTOMER CHURN PREDICTION APP")
print("="*60 + "\n")

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
print("[1/3] Loading model...")
with open('model_balanced.pkl', 'rb') as f:
    model = pickle.load(f)
print("✓ Model loaded successfully")

print("[2/3] Loading scaler...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Scaler loaded successfully")

print("[3/3] Flask app initialized")
print("\n" + "="*60 + "\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = float(request.form.get('age', 0))
        tenure = float(request.form.get('tenure', 0))
        monthly_charges = float(request.form.get('monthly_charges', 0))
        gender = request.form.get('gender', 'Male')
        
        # Encode gender (Male=1, Female=0)
        gender_encoded = 1 if gender == 'Male' else 0
        
        # Prepare features
        features = np.array([[age, tenure, monthly_charges, gender_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Format output based on prediction
        if prediction == 1:
            status = 'churn'
            output = f'🔴 Retention Risk ({probabilities[1]:.1%})'
        else:
            status = 'no-churn'
            output = f'🟢 Low Risk ({probabilities[0]:.1%})'
        
        return render_template('index.html', prediction_text=output, status=status)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}', 
                             status='error')

if __name__ == '__main__':
    print("🚀 Running on http://127.0.0.1:5000")
    print("📊 Press Ctrl+C to stop\n")
    app.run(debug=True, use_reloader=False)