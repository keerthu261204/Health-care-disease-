import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Title
st.title("Disease Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Handle missing values
    df.ffill(inplace=True)

    # Encode categorical features (excluding target)
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'disease':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Encode target if needed
    if df['disease'].dtype == 'object':
        le_target = LabelEncoder()
        df['disease'] = le_target.fit_transform(df['disease'])

    # Split features and target
    X = df.drop('disease', axis=1)
    y = df['disease']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: {acc:.2f}")

    st.write("### Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature importance
    st.write("### Feature Importances:")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    fig2, ax2 = plt.subplots()
    feature_importances.sort_values(ascending=False).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)
