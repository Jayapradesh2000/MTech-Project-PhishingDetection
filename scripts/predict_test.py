import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("model", "phishing_model.pkl")

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

if __name__ == "__main__":
    model = load_model()
    # Example sample: URL_Length, Has_IP, Prefix_Suffix
    # Change these numbers to test different inputs
    sample = np.array([[100, 0, 1]])  # shape (1, 3)
    pred = model.predict(sample)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample).max()
    print("Prediction (1=phishing, 0=legit):", int(pred[0]))
    if prob is not None:
        print("Confidence:", float(prob))
