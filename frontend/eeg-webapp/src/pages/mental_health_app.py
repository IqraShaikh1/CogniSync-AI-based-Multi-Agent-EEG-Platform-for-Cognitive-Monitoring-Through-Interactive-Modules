# mental_health_app.py
# Flask API – EXACT replica of working real-time EEG testing logic

import os 
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from collections import deque
from scipy.signal import welch
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================


# ================= PATH RESOLUTION (CRITICAL) =================


MODEL_PATH = "eeg_mental_health_model.keras"
SCALER_PATH = "mental_health_scaler.joblib"
LABEL_ENCODER_PATH = "mental_health_label_encoder.joblib"

SAMPLE_RATE = 256          # Must match React / device
WINDOW_SEC = 2             # Same as testing script
MODEL_WINDOW = 512         # Sequence length
FEATURE_DIM = 7
# =========================================

app = Flask(__name__)
CORS(app)

# ================= LOAD ARTIFACTS =================
print("Loading model and preprocessors...")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("✓ Model loaded")
print("✓ Scaler loaded")
print("✓ Label encoder loaded")
print("✓ Classes:", label_encoder.classes_)
print("✓ Model input shape:", model.input_shape)

MODEL_LOADED = True

# ================= FEATURE EXTRACTION =================
def compute_features(eeg_window):
    freqs, psd = welch(
        eeg_window,
        fs=SAMPLE_RATE,
        nperseg=min(256, len(eeg_window))
    )

    def band_power(low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.trapz(psd[mask], freqs[mask])

    alpha = band_power(8, 12)
    beta = band_power(13, 30)
    theta = band_power(4, 7)

    return [
        float(np.mean(eeg_window)),                 # eeg_value
        float(alpha),                               # alpha_power
        float(beta),                                # beta_power
        float(theta),                               # theta_power
        float(alpha / (beta + 1e-6)),               # alpha_beta_ratio
        float(np.var(eeg_window)),                  # signal_variance
        float(beta / (alpha + theta + 1e-6)),       # attention_index
    ]

# ================= API ROUTES =================
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "success",
        "model_loaded": MODEL_LOADED,
        "states": list(label_encoder.classes_),
        "expected_shape": "(1, 512, 7)"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not MODEL_LOADED:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json
        if "signal" not in data:
            return jsonify({"error": "Missing 'signal'"}), 400

        signal = np.asarray(data["signal"], dtype=np.float32)
        fs = int(data.get("fs", SAMPLE_RATE))

        required_samples = fs * WINDOW_SEC
        if len(signal) < required_samples:
            return jsonify({
                "error": f"At least {required_samples} samples required"
            }), 400

        # ---------- MATCH TESTING SCRIPT LOGIC ----------
        raw_buffer = deque(
            signal[-required_samples:], maxlen=required_samples
        )

        feature_sequence = deque(maxlen=MODEL_WINDOW)

        # Build feature sequence exactly like serial script
        while len(feature_sequence) < MODEL_WINDOW:
            features = compute_features(np.array(raw_buffer))
            feature_sequence.append(features)

        X = np.array(feature_sequence, dtype=np.float32)   # (512, 7)
        X_scaled = scaler.transform(X)                     # scale per timestep
        X_scaled = X_scaled.reshape(1, MODEL_WINDOW, FEATURE_DIM)

        preds = model.predict(X_scaled, verbose=0)[0]
        idx = int(np.argmax(preds))

        response = {
            "prediction": idx,
            "state_name": label_encoder.classes_[idx],
            "confidence": float(preds[idx]),
            "probabilities": {
                label_encoder.classes_[i]: float(preds[i])
                for i in range(len(preds))
            },
            "status": "success"
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    })

# ================= RUN =================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 EEG Mental Health Classification API")
    print("=" * 60)
    print("Server running at http://127.0.0.1:8000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=8000, debug=True)
