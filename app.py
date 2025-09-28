from flask import Flask, request, jsonify, render_template, render_template_string
import os
import joblib
import numpy as np
from utils.preprocessing import extract_features_from_url

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "phishing_model.pkl")

# --- Load model (support both dict-bundle and raw model) ---
model = None
FEATURE_NAMES = []

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        # If we saved a dict with model + feature_names in train script
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            FEATURE_NAMES = bundle.get("feature_names", [])
        else:
            # legacy: bundle might be the raw model object
            model = bundle
            # If feature names are not known, provide a sensible default
            FEATURE_NAMES = ["URL_Length", "Has_IP", "Prefix_Suffix"]
        print("Loaded model from:", MODEL_PATH)
        print("Feature names:", FEATURE_NAMES)
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print("Model file not found. Train first with scripts/train_model.py")

# --- Inline fallback HTML if templates/index.html is missing ---
FALLBACK_HTML = """
<!doctype html>
<title>Phishing Detection Tool</title>
<h2>Phishing Detection Tool</h2>
<form method="post" action="/predict">
  <label for="url">Enter URL:</label><br>
  <input type="text" id="url" name="url" size="80" required><br><br>
  <input type="submit" value="Predict">
</form>
{% if result %}
  <h3>Result</h3>
  <p>Prediction: <strong>{{ result.prediction }}</strong></p>
  <p>Confidence: {{ result.confidence }}</p>
  <pre>{{ result.features }}</pre>
{% endif %}
"""

def render_home(result=None):
    """
    Render using templates/index.html if present, otherwise fallback.
    Pass `result` (dict) to the template when available.
    """
    template_path = os.path.join(app.root_path, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html", result=result)
    else:
        return render_template_string(FALLBACK_HTML, result=result)

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return render_home(result=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded. Train and save a model first (scripts/train_model.py).", 500

    # Accept form submission or JSON { "url": "..." }
    if request.is_json:
        data = request.get_json(silent=True) or {}
        url = data.get("url")
    else:
        url = request.form.get("url")

    if not url:
        return "No URL provided", 400

    # Extract features and build sample in feature order
    feats = extract_features_from_url(url)

    # Ensure all FEATURE_NAMES exist in feats (fill missing with 0)
    sample_features = [feats.get(name, 0) for name in FEATURE_NAMES]
    sample = np.array([sample_features], dtype=float)

    try:
        pred_raw = model.predict(sample)[0]
        pred = int(pred_raw)
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "detail": str(e)}), 500

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(sample).max())
        except Exception:
            prob = None

    result = {
        "prediction": "phishing" if pred == 1 else "legitimate",
        "confidence": prob,
        "features": feats
    }

    if request.is_json:
        return jsonify(result)
    else:
        return render_home(result=result)

# Optional: JSON-only endpoint for programmatic use
@app.route("/predict_json", methods=["POST"])
def predict_json():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Provide JSON with key 'url'"}), 400

    url = data["url"]
    feats = extract_features_from_url(url)
    sample_features = [feats.get(name, 0) for name in FEATURE_NAMES]
    sample = np.array([sample_features], dtype=float)

    pred = int(model.predict(sample)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(sample).max())
        except Exception:
            prob = None

    return jsonify({
        "prediction": "phishing" if pred == 1 else "legitimate",
        "confidence": prob,
        "features": feats
    })

if __name__ == "__main__":
    # Use "python -m app" or "python app.py" to run
    app.run(debug=True)
