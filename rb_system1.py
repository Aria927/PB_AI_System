# rb_system1_expert.py
# Rule-Based System 1: Expert Rules (using only training_data.csv)

import pandas as pd
from sklearn.metrics import accuracy_score

# === Load dataset ===
data = pd.read_csv("training_data.csv")

# === Generate synthetic "true" labels for testing ===
def true_risk_label(row):
    """
    Synthetic rule to simulate true risk levels.
    """
    if row["Credit amount"] > 10000 or row["Duration"] > 40:
        return "high_risk"
    elif row["Credit amount"] < 2500 and row["Duration"] < 15:
        return "low_risk"
    else:
        return "medium_risk"

# Apply the synthetic true labels
data["True_Risk"] = data.apply(true_risk_label, axis=1)

# === Rule-based system prediction ===
def predict_risk(row):
    """
    Simulates expert-like rule predictions.
    """
    if row["Credit amount"] > 12000 or row["Duration"] > 45:
        return "high_risk"
    elif row["Credit amount"] < 2000 and row["Duration"] < 12:
        return "low_risk"
    else:
        return "medium_risk"

# Apply prediction rules
data["Predicted_Risk"] = data.apply(predict_risk, axis=1)

# === Evaluate ===
accuracy = accuracy_score(data["True_Risk"], data["Predicted_Risk"])

print("=== Rule-Based System 1: Expert Rules ===")
print(f"Accuracy on training_data.csv: {accuracy:.2f}")

# === Optional detailed breakdown ===
comparison = pd.crosstab(data["True_Risk"], data["Predicted_Risk"], rownames=["True"], colnames=["Predicted"])
print("\nConfusion Matrix:")
print(comparison)

# === Show some example predictions ===
print("\nSample predictions:")
print(data[["Credit amount", "Duration", "True_Risk", "Predicted_Risk"]].head(10))

