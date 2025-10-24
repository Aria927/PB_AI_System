# === Rule-Based AI System 2: Decision Tree-Based Rules ===
# This system automatically generates interpretable rules from the data.

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Load the training dataset ===
data = pd.read_csv("training_data.csv")

# Clean up column names
data.columns = data.columns.str.strip()

# --- OPTIONAL: Standardize labels to match System 1 naming convention ---
# If the Risk column contains "good"/"bad", convert it to "low_risk"/"high_risk"
if data["Risk"].str.contains("good|bad", case=False).any():
    data["Risk"] = data["Risk"].replace({
        "good": "low_risk",
        "bad": "high_risk"
    })

# === Separate features (X) and target (y) ===
X = data.drop(columns=["Risk", "ID"], errors="ignore")
y = data["Risk"]

# === Handle missing values ===
X = X.fillna("unknown")

# === Encode categorical features ===
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# === Encode target variable ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === Train a Decision Tree (interpretable rule-based model) ===
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_encoded, y_encoded)

# === Make predictions ===
y_pred = tree.predict(X_encoded)

# === Evaluate accuracy ===
accuracy = accuracy_score(y_encoded, y_pred)
print("=== Rule-Based System 2: Decision Tree Rules ===")
print(f"Accuracy on training_data.csv: {accuracy:.2f}")

# === Display the learned rules ===
rules = export_text(tree, feature_names=list(X.columns))
print("\nLearned Decision Tree Rules:\n")
print(rules)

# === Generate and display confusion matrix ===
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Rule-Based System 2 (Decision Tree)")
plt.show()

# Print confusion matrix in table format
print("\nConfusion Matrix:")
print(pd.DataFrame(cm,
                   index=label_encoder.classes_,
                   columns=label_encoder.classes_))

# === Add predictions to DataFrame for reference ===
data["Predicted_Risk"] = label_encoder.inverse_transform(y_pred)

print("\nSample Predictions:")
print(data[["Age", "Sex", "Credit amount", "Duration", "Housing", "Predicted_Risk"]].head(10))
