import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# =========================
# STEP 1: LOAD DATA
# =========================

# Apni CSV file ka exact naam yahan likho
df = pd.read_csv("T_ONTIME_REPORTING.csv", low_memory=False)

print("Original Shape:", df.shape)

# =========================
# STEP 2: CLEANING
# =========================

# Remove rows with missing arrival delay
df = df[df["ARR_DELAY"].notna()]

# Remove rows with missing departure delay
df = df[df["DEP_DELAY"].notna()]

# Remove rows with missing target
df = df[df["ARR_DEL15"].notna()]

print("After Cleaning Shape:", df.shape)

# =========================
# STEP 3: SELECT FEATURES
# =========================

df = df[[
    "MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "DISTANCE",
    "TAXI_OUT",
    "ARR_DEL15"
]]

print("Selected Columns:", df.columns)

# =========================
# STEP 4: ENCODING
# =========================

# One-hot encoding airline column
df = pd.get_dummies(df, columns=["OP_UNIQUE_CARRIER"], drop_first=True)

# =========================
# STEP 5: TRAIN TEST SPLIT
# =========================

X = df.drop("ARR_DEL15", axis=1)
y = df["ARR_DEL15"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =========================
# STEP 6: MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# STEP 7: EVALUATION
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_score)

# =========================
# STEP 8: SAVE MODEL
# =========================

# Save feature names
joblib.dump(model, "flight_delay_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("\nModel Saved Successfully as flight_delay_model.pkl")

import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = X.columns

feat_importance = pd.Series(importances, index=feature_names)
feat_importance = feat_importance.sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(feat_importance.head(10))

# Plot
feat_importance.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()