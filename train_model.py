"""
Train Customer Churn Prediction Model
This script generates sample customer data and trains a churn prediction model
with proper feature scaling and model persistence.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("="*60)
print("CUSTOMER CHURN PREDICTION MODEL TRAINING")
print("="*60)

# Create synthetic customer dataset
print("\n1. Generating synthetic customer data...")
n_samples = 1000

# Generate features
ages = np.random.randint(18, 80, n_samples)
tenures = np.random.randint(0, 73, n_samples)  # 0-6 years
monthly_charges = np.random.uniform(20, 120, n_samples)
genders = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male

# Create churn target based on some patterns
# Higher tenure = less churn, higher charges = more churn
churn_prob = (0.3 * (1 - tenures/72) +  # Newer customers more likely to churn
              0.2 * (monthly_charges/120) +  # Higher charges = more churn
              0.1 * genders +  # Slight gender difference
              np.random.normal(0, 0.1, n_samples))  # Random noise

churn = (churn_prob > 0.5).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'age': ages,
    'tenure': tenures,
    'monthly_charges': monthly_charges,
    'gender': genders,
    'churn': churn
})

print(f"   ✓ Generated {n_samples} customer records")
print(f"   ✓ Churn rate: {churn.mean():.1%}")
print(f"\nDataset Summary:")
print(data.describe())

# Prepare features and target
print("\n2. Preparing features and target...")
X = data[['age', 'tenure', 'monthly_charges', 'gender']].values
y = data['churn'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# Scale features
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ StandardScaler fitted and applied")

# Train model
print("\n4. Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train_scaled, y_train)
print("   ✓ Model training completed")

# Evaluate model
print("\n5. Model Evaluation:")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"   ✓ Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Save scaler
print("\n6. Saving scaler...")
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler saved to scaler.pkl")

# Save model as model.pkl
print("\n7. Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved to model.pkl")

# Save model as model_balanced.pkl (same model)
with open('model_balanced.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model also saved to model_balanced.pkl")

# Verify files
print("\n8. Verifying saved files...")
import os
files = ['scaler.pkl', 'model.pkl', 'model_balanced.pkl']
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✓ {file}: {size:,} bytes")
    else:
        print(f"   ✗ {file}: NOT FOUND")

# Test loading
print("\n9. Testing pickle file loading...")
try:
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    print("   ✓ All files loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading files: {e}")

print("\n" + "="*60)
print("TRAINING COMPLETE - Ready for Flask app!")
print("="*60)

# Test prediction
print("\n10. Test Prediction:")
test_sample = np.array([[35, 12, 65.50, 1]])  # Age=35, Tenure=12, Charges=65.50, Gender=Male
test_scaled = loaded_scaler.transform(test_sample)
pred = loaded_model.predict(test_scaled)[0]
prob = loaded_model.predict_proba(test_scaled)[0]
print(f"    Input: Age=35, Tenure=12mo, Monthly=$65.50, Gender=Male")
print(f"    Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
print(f"    Confidence: {prob[pred]:.1%}")
