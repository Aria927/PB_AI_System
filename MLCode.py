# 1. Import Required Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the Dataset
df = pd.read_csv("training_data.csv") 

print("Dataset Loaded:")
print(df.head())
print("\nDataset Shape:", df.shape)

# 3. Basic Data Inspection
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 4. Handle Missing Data
# Simple strategy: numeric columns → fill mean, categorical → fill mode
for col in df.columns:
    if df[col].dtype in ("float64", "int64"):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 5. Define and Encode Target
TARGET = "Risk"

# Encode the Risk column before dummy variables
df[TARGET] = df[TARGET].map({"good": 0, "bad": 1})

# 6. Convert All Other Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# Split into Input & Output
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 7. Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42
)

print("\nTrain Size:", X_train.shape)
print("Test Size:", X_test.shape)

# 8. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Training Complete.")


# 10. Predict on Test Data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 11. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n=== Logistic Regression Performance ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")


# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# 13. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")  # Baseline
plt.title("Receiver Operating Characteristic – Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()