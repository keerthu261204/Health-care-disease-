import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load Dataset
data_path = r'C:\Users\DELL\Documents\sample_disease_dataset.csv'
df = pd.read_csv(data_path)
print("Dataset Shape:", df.shape)
print(df.head())

# Step 2: Handle Missing Values (updated to avoid FutureWarning)
df.ffill(inplace=True)  # Forward fill for missing values

# Step 3: Encode Categorical Columns (excluding target if needed)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'disease':  # Ensure we encode target separately if needed
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Step 4: Feature & Target Split
X = df.drop('disease', axis=1)  # Change if your target column name is different
y = df['disease']

# Step 5: Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train a Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
