import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Streamlit Page Setup
st.set_page_config(page_title="Disease Prediction App", layout="wide")
st.title("Disease Prediction using Random Forest")
st.markdown("Upload your dataset and get evaluation metrics instantly.")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())
    st.write("Dataset Shape:", df.shape)

    # Step 3: Handle Missing Values
    df.ffill(inplace=True)

    # Step 4: Encode Categorical Features
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'disease':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Step 5: Encode Target
    y = df['disease']
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Step 6: Feature Selection & Scaling
    X = df.drop('disease', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 7: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 8: Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 9: Model Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text("Classification Report:\n" + class_report)

    # Step 10: Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("Model training and evaluation complete.")

else:
    st.info("Please upload a CSV file to get started.")
