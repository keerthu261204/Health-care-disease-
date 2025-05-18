import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Manual Disease Predictor", layout="centered")
st.title("Disease Prediction from Symptoms")

# Sample Dataset (can be replaced with real medical dataset)
# Assume dataset with features: age, temperature, cough, fatigue
data = {
    'age': [25, 45, 30, 60, 22, 36, 52, 47],
    'temperature': [98.6, 101.4, 99.2, 102.1, 98.7, 100.5, 101.0, 99.8],
    'cough': [1, 1, 0, 1, 0, 1, 1, 0],
    'fatigue': [0, 1, 0, 1, 0, 1, 1, 1],
    'disease': ['None', 'Flu', 'None', 'COVID-19', 'None', 'Flu', 'COVID-19', 'Flu']
}
df = pd.DataFrame(data)

# Encode target
le = LabelEncoder()
df['disease_encoded'] = le.fit_transform(df['disease'])

# Features and Target
X = df[['age', 'temperature', 'cough', 'fatigue']]
y = df['disease_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User Inputs
st.subheader("Enter Your Symptoms")

age = st.slider("Age", 0, 100, 30)
temperature = st.slider("Body Temperature (°F)", 95.0, 105.0, 98.6)
cough = st.selectbox("Cough", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])

# Convert to numeric
cough_val = 1 if cough == "Yes" else 0
fatigue_val = 1 if fatigue == "Yes" else 0

# Make prediction
input_data = np.array([[age, temperature, cough_val, fatigue_val]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
predicted_disease = le.inverse_transform(prediction)[0]

# Show result
st.subheader("Predicted Disease")
st.write(f"Based on your symptoms, the model predicts: **{predicted_disease}**")
