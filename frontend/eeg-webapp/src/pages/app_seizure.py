# seizure_app.py - Flask API for Seizure Detection
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from scipy.signal import welch
import warnings
import io
import pandas as pd
import os
import sys
import requests
# Windows UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ===============================
# Load Model and Scaler
# ===============================
MODEL_LOADED = False
model = None
scaler = None

try:
    print("Loading seizure model and scaler...")

    model = joblib.load("seizure_xgb_focus.pkl")
    print("✅ Loaded: seizure_xgb_focus.pkl")

    scaler = joblib.load("seizure_focus_scaler.pkl")
    print("✅ Loaded: seizure_focus_scaler.pkl")

    MODEL_LOADED = True
    print("\n✅ All components loaded successfully!\n")

except Exception as e:
    print(f"\n❌ Error loading model/scaler: {e}")
    print("Please make sure seizure_xgb_focus.pkl and seizure_focus_scaler.pkl are in the same directory!\n")
    MODEL_LOADED = False


# ===============================
# Feature Extraction (FOCUS MODULE STANDARD)
# Matches EXACTLY the training script's extract_focus_features()
# ===============================
def extract_focus_features(eeg_window, fs=250):
    """
    Extract the 6 features used during training.
    Must match training script exactly.
    """
    freqs, psd = welch(eeg_window, fs=fs, nperseg=256)

    alpha_mask = (freqs >= 8)  & (freqs <= 12)
    beta_mask  = (freqs >= 13) & (freqs <= 30)
    theta_mask = (freqs >= 4)  & (freqs <= 7)

    alpha_p = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
    beta_p  = np.trapezoid(psd[beta_mask],  freqs[beta_mask])
    theta_p = np.trapezoid(psd[theta_mask], freqs[theta_mask])

    alpha_beta_ratio = alpha_p / (beta_p + 1e-10)
    signal_variance  = np.var(eeg_window)
    attention_index  = beta_p / (alpha_p + theta_p + 1e-10)

    return {
        "alpha_power":       float(alpha_p),
        "beta_power":        float(beta_p),
        "theta_power":       float(theta_p),
        "alpha_beta_ratio":  float(alpha_beta_ratio),
        "signal_variance":   float(signal_variance),
        "attention_index":   float(attention_index),
    }


# ===============================
# API Endpoints
# ===============================

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify API is running"""
    return jsonify({
        'status': 'success',
        'message': 'Seizure Detection API is running',
        'model_loaded': MODEL_LOADED,
        'model_type': 'XGBoost Seizure Classifier'
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': str(np.datetime64('now'))
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Accepts JSON with either:
      1. Raw signal:
         { "signal": [float, ...], "fs": int }

      2. Pre-computed focus features (matching focus2.csv columns):
         {
           "alpha_power": float,
           "beta_power": float,
           "theta_power": float,
           "alpha_beta_ratio": float,
           "signal_variance": float,
           "attention_index": float
         }

    Returns:
      {
        "prediction":      0 | 1,          # 0 = No Seizure, 1 = Seizure
        "label":           "No Seizure" | "Seizure",
        "seizure_prob":    float,           # 0.0 – 1.0
        "normal_prob":     float,
        "alert_level":     "LOW" | "MEDIUM" | "HIGH",
        "features":        { ... },
        "model_type":      str,
        "status":          "success"
      }
    """
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded. Make sure seizure_xgb_focus.pkl and seizure_focus_scaler.pkl are present!'
            }), 500

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON body received'}), 400

        # ------------------------------------------------------------------
        # Feature extraction
        # ------------------------------------------------------------------
        if 'signal' in data:
            signal = np.array(data['signal'], dtype=np.float64)
            fs     = int(data.get('fs', 250))

            if len(signal) < 256:
                return jsonify({'error': f'Signal too short: need ≥ 256 samples, got {len(signal)}'}), 400

            features = extract_focus_features(signal, fs=fs)
            print(f"📊 Features extracted from {len(signal)} samples at {fs} Hz")

        else:
            required = ["alpha_power", "beta_power", "theta_power",
                        "alpha_beta_ratio", "signal_variance", "attention_index"]
            missing = [k for k in required if k not in data]
            if missing:
                return jsonify({'error': f'Missing feature keys: {missing}'}), 400

            features = {k: float(data[k]) for k in required}
            print("📊 Features received directly from request")

        # ------------------------------------------------------------------
        # Build feature vector (same order as training)
        # ------------------------------------------------------------------
        FEATURE_ORDER = [
            "alpha_power", "beta_power", "theta_power",
            "alpha_beta_ratio", "signal_variance", "attention_index"
        ]
        X = np.array([[features[k] for k in FEATURE_ORDER]])

        X_scaled = scaler.transform(X)

        prediction    = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]

        normal_prob  = float(probabilities[0])
        seizure_prob = float(probabilities[1])

        label = "Seizure" if prediction == 1 else "No Seizure"
       
        try:
            duration_s = len(signal) / fs if 'signal' in data else 2.0  # fallback 2 sec

            on_seizure_prediction_complete(
            duration_s = duration_s,
            detected   = (prediction == 1),
            confidence = seizure_prob,
            seizure_type = ""   # You can improve later
            )
            print("💾 Seizure session saved to Journal DB")
        except Exception as db_error:
            print(f"⚠️ Could not save seizure session: {db_error}")


        if seizure_prob >= 0.75:
            alert_level = "HIGH"
        elif seizure_prob >= 0.40:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"

        print(f"🎯 Prediction: {label} | Seizure prob: {seizure_prob:.2%} | Alert: {alert_level}")

        return jsonify({
            'prediction':   prediction,
            'label':        label,
            'seizure_prob': seizure_prob,
            'normal_prob':  normal_prob,
            'alert_level':  alert_level,
            'features':     features,
            'model_type':   'XGBoost Seizure Classifier',
            'status':       'success'
        })

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'details': 'Check server logs for more information'
        }), 500


@app.route('/predict_window', methods=['POST'])
def predict_window():
    """
    Batch window endpoint — mirrors testing.py logic.

    Accepts:
      {
        "windows": [
          {
            "timestamp_ms": int,
            "alpha_power": float,
            "beta_power": float,
            "theta_power": float,
            "alpha_beta_ratio": float,
            "signal_variance": float,
            "attention_index": float
          },
          ...
        ]
      }

    Returns per-window predictions + overall session decision.
    """
    try:
        if not MODEL_LOADED:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        windows = data.get('windows', [])
        if not windows:
            return jsonify({'error': 'No windows provided'}), 400

        FEATURE_ORDER = [
            "alpha_power", "beta_power", "theta_power",
            "alpha_beta_ratio", "signal_variance", "attention_index"
        ]

        X_raw = np.array([[w[k] for k in FEATURE_ORDER] for w in windows])
        X_scaled = scaler.transform(X_raw)

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        window_results = []
        for i, w in enumerate(windows):
            window_results.append({
                'timestamp_ms': w.get('timestamp_ms', i),
                'pred_label':   int(y_pred[i]),
                'label':        "Seizure" if y_pred[i] == 1 else "No Seizure",
                'seizure_prob': float(y_prob[i])
            })

        total_windows   = len(y_pred)
        seizure_windows = int(np.sum(y_pred == 1))
        normal_windows  = int(np.sum(y_pred == 0))
        seizure_ratio   = seizure_windows / total_windows
        avg_prob        = float(np.mean(y_prob))
        final_label     = "SEIZURE DETECTED" if seizure_ratio >= 0.30 else "NORMAL"

        return jsonify({
            'window_results':   window_results,
            'total_windows':    total_windows,
            'seizure_windows':  seizure_windows,
            'normal_windows':   normal_windows,
            'seizure_ratio':    seizure_ratio,
            'avg_seizure_prob': avg_prob,
            'final_decision':   final_label,
            'status':           'success'
        })

    except Exception as e:
        print(f"❌ Batch predict error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ===============================
# NEW: CSV Upload Endpoint
# Mirrors testing.py logic exactly
# ===============================
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    CSV upload endpoint — replicates testing.py pipeline exactly.

    Accepts a multipart/form-data POST with:
      - file : the CSV file  (required)
      - window_samples : int (optional, default 500 = 2s @ 250Hz)

    The CSV must have these columns (matching focus2.csv format):
      timestamp_ms, eeg_value, alpha_power, beta_power, theta_power,
      alpha_beta_ratio, signal_variance, attention_index

    If timestamp_ms is absent, row index × 4 ms is used.

    Returns full per-window table + session summary matching testing.py output:
    {
      "window_results": [
        { "timestamp_ms": int, "pred_label": 0|1,
          "label": "No Seizure"|"Seizure", "seizure_prob": float },
        ...
      ],
      "summary": {
        "total_windows":    int,
        "seizure_windows":  int,
        "normal_windows":   int,
        "seizure_ratio":    float,
        "avg_seizure_prob": float,
        "final_decision":   "SEIZURE DETECTED" | "NORMAL"
      },
      "status": "success"
    }
    """
    try:
        if not MODEL_LOADED:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Send multipart/form-data with key "file"'}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400

        window_samples = int(request.form.get('window_samples', 500))   # default 500 (2s @ 250Hz)

        # Read CSV
        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        print(f"📂 CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

        FEATURE_COLS = [
            "alpha_power", "beta_power", "theta_power",
            "alpha_beta_ratio", "signal_variance", "attention_index"
        ]

        # Validate feature columns exist
        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {missing_cols}',
                'found_columns': list(df.columns),
                'required_columns': FEATURE_COLS
            }), 400

        # timestamp_ms column (optional)
        has_timestamp = 'timestamp_ms' in df.columns

        # -------------------------------------------------------
        # Window loop — exactly mirrors testing.py
        # -------------------------------------------------------
        windows      = []
        timestamps   = []

        for i in range(0, len(df) - window_samples + 1, window_samples):
            window          = df.iloc[i:i + window_samples]
            window_features = window[FEATURE_COLS].mean().values   # aggregate features (mean)
            windows.append(window_features)

            if has_timestamp:
                timestamps.append(int(window['timestamp_ms'].iloc[0]))
            else:
                timestamps.append(i * 4)   # approximate ms offset

        if not windows:
            return jsonify({
                'error': f'Not enough rows for even one window. Need ≥ {window_samples} rows, got {len(df)}'
            }), 400

        X_windows = np.array(windows)
        X_scaled  = scaler.transform(X_windows)

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # -------------------------------------------------------
        # Per-window results table
        # -------------------------------------------------------
        window_results = []
        for i in range(len(y_pred)):
            window_results.append({
                'timestamp_ms': timestamps[i],
                'pred_label':   int(y_pred[i]),
                'label':        "Seizure" if y_pred[i] == 1 else "No Seizure",
                'seizure_prob': round(float(y_prob[i]), 6)
            })

        # -------------------------------------------------------
        # Session summary — mirrors testing.py console output
        # -------------------------------------------------------
        total_windows   = len(y_pred)
        seizure_windows = int(np.sum(y_pred == 1))
        normal_windows  = int(np.sum(y_pred == 0))
        seizure_ratio   = round(seizure_windows / total_windows, 4)
        avg_prob        = round(float(np.mean(y_prob)), 6)
        final_decision  = "SEIZURE DETECTED" if seizure_ratio >= 0.30 else "NORMAL"
      
        try:
            session_duration_s = len(df) / 250  # assuming 250 Hz
            detected_flag = final_decision == "SEIZURE DETECTED"

            on_seizure_prediction_complete(
            duration_s  = session_duration_s,
            detected    = detected_flag,
            confidence  = avg_prob,
            seizure_type = ""  # optional
            )
            print("💾 CSV seizure session saved to Journal DB")
        except Exception as db_error:
            print(f"⚠️ Could not save CSV seizure session: {db_error}")


        # Console output matching testing.py exactly
        print("\n================ FINAL EEG PREDICTION ================")
        print(f"Total windows analyzed : {total_windows}")
        print(f"Seizure windows        : {seizure_windows}")
        print(f"Normal windows         : {normal_windows}")
        print(f"Seizure ratio          : {seizure_ratio:.2f}")
        print(f"Avg seizure probability: {avg_prob:.2f}")
        print(f"\n🧠 FINAL DECISION: {final_decision}")
        print("======================================================\n")

        return jsonify({
            'window_results': window_results,
            'summary': {
                'total_windows':    total_windows,
                'seizure_windows':  seizure_windows,
                'normal_windows':   normal_windows,
                'seizure_ratio':    seizure_ratio,
                'avg_seizure_prob': avg_prob,
                'final_decision':   final_decision
            },
            'csv_info': {
                'filename':       file.filename,
                'total_rows':     len(df),
                'window_samples': window_samples,
                'columns_used':   FEATURE_COLS
            },
            'status': 'success'
        })

    except Exception as e:
        print(f"❌ CSV predict error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'details': 'Check server logs for more information'
        }), 500

def on_seizure_prediction_complete(duration_s, detected, confidence, seizure_type=""):
    requests.post("http://localhost:5000/api/sessions/seizure", json={
        "duration_s":    duration_s,
        "detected":      detected,       # True / False
        "confidence":    confidence,     # float, 0.0–1.0
        "seizure_type":  seizure_type,   # "Focal" | "Generalized" | ""
        "notes":         ""
    })
# ===============================
# Entry Point
# ===============================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 Seizure Detection API Server - XGBoost Model")
    print("="*60)
    print(f"Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Loaded'}")
    if MODEL_LOADED:
        print("Model Type: XGBoost (Bonn Dataset, Focus Feature Standard)")
        print("Features  : alpha_power, beta_power, theta_power,")
        print("            alpha_beta_ratio, signal_variance, attention_index")
    print("="*60)
    print("Starting server on http://127.0.0.1:8001")
    print("\nEndpoints:")
    print("  GET  /test            - Test if API is running")
    print("  GET  /health          - Health check")
    print("  POST /predict         - Single window prediction")
    print("  POST /predict_window  - Batch window prediction")
    print("  POST /predict_csv     - CSV file upload & batch prediction")
    print("="*60 + "\n")

    port = int(os.environ.get('PORT', 8001))
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)