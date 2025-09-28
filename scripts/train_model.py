import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.preprocessing import extract_features_from_url

# Paths
DATA_PATH = os.path.join("data", "dataset_urls_clean.csv")
MODEL_PATH = os.path.join("model", "phishing_model.pkl")

# Load dataset (url + label)
df = pd.read_csv(DATA_PATH)

# Extract features for each URL
feature_rows = []
for _, row in df.iterrows():
    feats = extract_features_from_url(row["url"])
    feats["label"] = row["label"]
    feature_rows.append(feats)

# Convert to DataFrame
feature_df = pd.DataFrame(feature_rows)

# Features and labels
X = feature_df.drop("label", axis=1)
y = feature_df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model + feature names
joblib.dump({"model": model, "feature_names": list(X.columns)}, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
