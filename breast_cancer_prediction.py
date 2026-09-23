"""
CodeAlpha - Task 4: Disease Prediction from Medical Data
Project: Breast Cancer Prediction
Dataset: Wisconsin Breast Cancer Dataset (built into scikit-learn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# ----------------------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

print("Dataset shape:", df.shape)
print(df["target"].value_counts())
print(df.head())

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# Check for missing values
print("\nMissing values:\n", X.isnull().sum().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. Model Training
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "ROC-AUC": auc
    }

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

# ----------------------------------------------------------------------
# 4. Compare Models
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df)

results_df.plot(kind="bar", figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ----------------------------------------------------------------------
# 5. Confusion Matrix (Best Model: Random Forest)
# ----------------------------------------------------------------------
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ----------------------------------------------------------------------
# 6. ROC Curve
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# ----------------------------------------------------------------------
# 7. Feature Importance (Random Forest) - Bonus for standout project
# ----------------------------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh")
plt.title("Top 10 Important Features - Random Forest")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nTop 10 features influencing prediction:\n", importances)

# ----------------------------------------------------------------------
# 8. Save Best Model (optional, for deployment)
# ----------------------------------------------------------------------
import joblib
joblib.dump(best_model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler saved successfully.")