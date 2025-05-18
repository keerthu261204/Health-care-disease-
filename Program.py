import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("Disease Prediction from Symptoms")

# Expanded dataset (mock data)
data = {
    'age': [25, 45, 30, 60, 22, 36, 52, 47, 33, 40, 29, 50, 39, 28],
    'temperature': [98.6, 101.4, 99.2, 102.1, 98.7, 100.5, 101.0, 99.8, 100.1, 103.2, 97.9, 101.6, 102.3, 98.4],
    'cough':       [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    'fatigue':     [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    'headache':    [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    'sore_throat': [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    'body_pain':   [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    'disease': [
        'None', 'Flu', 'None', 'COVID-19', 'None', 'Flu', 'COVID-19', 'None',
        'Dengue', 'Malaria', 'None', 'Typhoid', 'Dengue', 'None'
    ]
}
df = pd.DataFrame(data)

# Encode target
le = LabelEncoder()
df['disease_encoded'] = le.fit_transform(df['disease'])

# Features & target
X = df.drop(['disease', 'disease_encoded'], axis=1)
y = df['disease_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- User Input ---
st.subheader("Enter Your Symptoms")

age = st.slider("Age", 1, 100, 30)
temperature = st.slider("Temperature (°F)", 95.0, 105.0, 98.6)
cough = st.selectbox("Cough", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
headache = st.selectbox("Headache", ["No", "Yes"])
sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])
body_pain = st.selectbox("Body Pain", ["No", "Yes"])

# Convert to numeric
cough_val = 1 if cough == "Yes" else 0
fatigue_val = 1 if fatigue == "Yes" else 0
headache_val = 1 if headache == "Yes" else 0
sore_throat_val = 1 if sore_throat == "Yes" else 0
body_pain_val = 1 if body_pain == "Yes" else 0

# Predict on button click
if st.button("Predict"):
    input_data = np.array([[age, temperature, cough_val, fatigue_val, headache_val, sore_throat_val, body_pain_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_disease = le.inverse_transform(prediction)[0]

    st.subheader("Predicted Disease")
    st.success(f"The model predicts you may have: **{predicted_disease}**")
