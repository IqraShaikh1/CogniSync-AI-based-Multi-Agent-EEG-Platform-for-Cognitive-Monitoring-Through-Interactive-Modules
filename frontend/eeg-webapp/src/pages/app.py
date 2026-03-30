# app.py - Flask API for EEG Classification
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from scipy import stats
import warnings
import os
import sys
import requests
import csv
import io

# Windows UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ===============================
# Load Model and Preprocessing Objects
# ===============================
MODEL_LOADED = False
model = None
scaler = None
feature_selector = None
variance_selector = None
selected_features = None

try:
    print("Loading model and preprocessing objects...")
    
    model = joblib.load("best_model_stacking_H.pkl")
    print("✅ Loaded: best_model_stacking_H.pkl")
    
    scaler = joblib.load("scaler.pkl")
    print("✅ Loaded: scaler.pkl")
    
    feature_selector = joblib.load("feature_selector.pkl")
    print("✅ Loaded: feature_selector.pkl")
    
    variance_selector = joblib.load("variance_selector.pkl")
    print("✅ Loaded: variance_selector.pkl")
    
    selected_features = joblib.load("selected_features.pkl")
    print("✅ Loaded: selected_features.pkl")
    
    MODEL_LOADED = True
    print("\n✅ All components loaded successfully!\n")
    
except Exception as e:
    print(f"\n❌ Error loading model/preprocessing objects: {e}")
    print("Please run 'python save_preprocessing.py' first!\n")
    MODEL_LOADED = False

# ===============================
# Feature Extraction Functions
# ===============================
def compute_band_power_features(band_powers):
    """Extract statistical features from band powers"""
    features = {}
    
    alpha = band_powers.get('alpha', 0)
    beta = band_powers.get('beta', 0)
    theta = band_powers.get('theta', 0)
    delta = band_powers.get('delta', 0)
    
    features['theta_beta_ratio'] = theta / (beta + 1e-6)
    features['alpha_theta_ratio'] = alpha / (theta + 1e-6)
    features['alpha_plus_theta_beta'] = (alpha + theta) / (beta + 1e-6)
    features['alpha_beta_ratio_mean'] = alpha / (beta + 1e-6)
    
    features['alpha_power_mean'] = alpha
    features['alpha_power_std'] = alpha * 0.1
    features['alpha_power_min'] = alpha * 0.5
    features['alpha_power_max'] = alpha * 1.5
    features['alpha_power_skew'] = 0.0
    features['alpha_power_kurt'] = 0.0
    
    features['beta_power_mean'] = beta
    features['beta_power_std'] = beta * 0.1
    features['beta_power_min'] = beta * 0.5
    features['beta_power_max'] = beta * 1.5
    features['beta_power_skew'] = 0.0
    features['beta_power_kurt'] = 0.0
    
    features['theta_power_mean'] = theta
    features['theta_power_std'] = theta * 0.1
    features['theta_power_min'] = theta * 0.5
    features['theta_power_max'] = theta * 1.5
    features['theta_power_skew'] = 0.0
    features['theta_power_kurt'] = 0.0
    
    features['alpha_beta_ratio_std'] = 0.0
    features['alpha_beta_ratio_min'] = features['alpha_beta_ratio_mean'] * 0.8
    features['alpha_beta_ratio_max'] = features['alpha_beta_ratio_mean'] * 1.2
    features['alpha_beta_ratio_skew'] = 0.0
    features['alpha_beta_ratio_kurt'] = 0.0
    
    total_power = alpha + beta + theta + delta
    features['signal_variance_mean'] = total_power
    features['signal_variance_std'] = total_power * 0.1
    features['signal_variance_min'] = total_power * 0.5
    features['signal_variance_max'] = total_power * 1.5
    features['signal_variance_skew'] = 0.0
    features['signal_variance_kurt'] = 0.0
    
    attention_idx = beta / (theta + alpha + 1e-6)
    features['attention_index_mean'] = attention_idx
    features['attention_index_std'] = attention_idx * 0.1
    features['attention_index_min'] = attention_idx * 0.8
    features['attention_index_max'] = attention_idx * 1.2
    features['attention_index_skew'] = 0.0
    features['attention_index_kurt'] = 0.0
    
    features['sample_entropy'] = 0.5
    features['perm_entropy'] = 0.5
    
    features['hjorth_activity'] = total_power
    features['hjorth_mobility'] = 1.0
    features['hjorth_complexity'] = 1.0
    
    return features

def extract_features_from_raw_signal(signal, fs=256):
    """Extract all 43 features from raw EEG signal"""
    signal = np.array(signal)
    
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(n, 1/fs)
    psd = np.abs(fft_vals) ** 2
    
    def band_power(lo, hi):
        idx = np.where((fft_freq >= lo) & (fft_freq < hi))
        return np.sum(psd[idx])
    
    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 12)
    beta = band_power(12, 30)
    
    features = {}
    
    for band_name, band_val in [('alpha', alpha), ('beta', beta), ('theta', theta)]:
        features[f'{band_name}_power_mean'] = band_val
        features[f'{band_name}_power_std'] = band_val * 0.15
        features[f'{band_name}_power_min'] = band_val * 0.5
        features[f'{band_name}_power_max'] = band_val * 1.5
        features[f'{band_name}_power_skew'] = 0.0
        features[f'{band_name}_power_kurt'] = 0.0
    
    features['theta_beta_ratio'] = theta / (beta + 1e-6)
    features['alpha_theta_ratio'] = alpha / (theta + 1e-6)
    features['alpha_plus_theta_beta'] = (alpha + theta) / (beta + 1e-6)
    features['alpha_beta_ratio_mean'] = alpha / (beta + 1e-6)
    features['alpha_beta_ratio_std'] = 0.0
    features['alpha_beta_ratio_min'] = features['alpha_beta_ratio_mean'] * 0.8
    features['alpha_beta_ratio_max'] = features['alpha_beta_ratio_mean'] * 1.2
    features['alpha_beta_ratio_skew'] = 0.0
    features['alpha_beta_ratio_kurt'] = 0.0
    
    variance = np.var(signal)
    features['signal_variance_mean'] = variance
    features['signal_variance_std'] = variance * 0.1
    features['signal_variance_min'] = variance * 0.5
    features['signal_variance_max'] = variance * 1.5
    try:
        features['signal_variance_skew'] = stats.skew(signal)
        features['signal_variance_kurt'] = stats.kurtosis(signal)
    except:
        features['signal_variance_skew'] = 0.0
        features['signal_variance_kurt'] = 0.0
    
    attention_idx = beta / (theta + alpha + 1e-6)
    features['attention_index_mean'] = attention_idx
    features['attention_index_std'] = attention_idx * 0.1
    features['attention_index_min'] = attention_idx * 0.8
    features['attention_index_max'] = attention_idx * 1.2
    features['attention_index_skew'] = 0.0
    features['attention_index_kurt'] = 0.0
    
    features['sample_entropy'] = 0.5
    features['perm_entropy'] = 0.5
    
    features['hjorth_activity'] = np.var(signal)
    diff1 = np.diff(signal)
    features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(signal) + 1e-6))
    diff2 = np.diff(diff1)
    features['hjorth_complexity'] = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-6)) / (features['hjorth_mobility'] + 1e-6)
    
    return features

def run_prediction_pipeline(features):
    """
    Shared helper: takes a features dict → runs full preprocessing pipeline
    → returns prediction, probabilities, focus_level.
    Used by both /predict and /predict-csv.
    """
    feature_names = [
        'theta_beta_ratio', 'alpha_theta_ratio', 'alpha_plus_theta_beta',
        'alpha_power_mean', 'alpha_power_std', 'alpha_power_min',
        'alpha_power_max', 'alpha_power_skew', 'alpha_power_kurt',
        'beta_power_mean', 'beta_power_std', 'beta_power_min',
        'beta_power_max', 'beta_power_skew', 'beta_power_kurt',
        'theta_power_mean', 'theta_power_std', 'theta_power_min',
        'theta_power_max', 'theta_power_skew', 'theta_power_kurt',
        'alpha_beta_ratio_mean', 'alpha_beta_ratio_std',
        'alpha_beta_ratio_min', 'alpha_beta_ratio_max',
        'alpha_beta_ratio_skew', 'alpha_beta_ratio_kurt',
        'signal_variance_mean', 'signal_variance_std',
        'signal_variance_min', 'signal_variance_max',
        'signal_variance_skew', 'signal_variance_kurt',
        'attention_index_mean', 'attention_index_std',
        'attention_index_min', 'attention_index_max',
        'attention_index_skew', 'attention_index_kurt',
        'sample_entropy', 'perm_entropy',
        'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
    ]

    import pandas as pd

    X = np.array([[features.get(name, 0.0) for name in feature_names]])

    X_var = variance_selector.transform(X)
    kept_features = [name for name, keep in zip(feature_names, variance_selector.get_support()) if keep]
    X_df = pd.DataFrame(X_var, columns=kept_features)
    X_sel = X_df[selected_features]
    X_scaled = scaler.transform(X_sel)
    X_final = feature_selector.transform(X_scaled)

    prediction = model.predict(X_final)[0]

    try:
        probabilities = model.predict_proba(X_final)[0]
        class_names = model.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(class_names, probabilities)}
    except:
        prob_dict = None

    if prob_dict:
        focus_prob = prob_dict.get('1', 0) + prob_dict.get('2', 0)
        distracted_prob = prob_dict.get('3', 0)
        focus_level = (focus_prob * 100) - (distracted_prob * 20)
        focus_level = max(0, min(100, focus_level))
    else:
        focus_level = (features.get('attention_index_mean', 0) /
                       (features.get('attention_index_mean', 0) + 1e-6)) * 50
        focus_level = max(0, min(100, focus_level))

    return str(prediction), prob_dict, float(focus_level)

# ===============================
# API Endpoints
# ===============================
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'EEG Classification API is running',
        'model_loaded': MODEL_LOADED,
        'model_type': 'Stacking Classifier'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint (live serial data).
    Accepts band powers or raw signal JSON.
    """
    try:
        if not MODEL_LOADED:
            return jsonify({'error': 'Model not loaded. Please run save_preprocessing.py first!'}), 500

        data = request.json

        if 'signal' in data:
            signal = data['signal']
            fs = data.get('fs', 256)
            features = extract_features_from_raw_signal(signal, fs)
            print(f"📊 Extracted features from {len(signal)} samples at {fs}Hz")
        else:
            features = compute_band_power_features(data)
            print("📊 Extracted features from band powers")

        prediction, prob_dict, focus_level = run_prediction_pipeline(features)

        print(f"🎯 Prediction: {prediction} | Focus: {focus_level:.1f}%")

        # ── Auto-save to Brain Journal DB ──────────────────────────────
        try:
            focus_score = focus_level / 100.0
            focus_label = "High" if focus_score >= 0.7 else ("Medium" if focus_score >= 0.4 else "Low")
            duration_s = len(data['signal']) / data.get('fs', 256) if 'signal' in data else 2.0
            on_focus_prediction_complete(duration_s, focus_score, focus_label)
            print("💾 Focus session saved to Journal DB")
        except Exception as db_error:
            print(f"⚠️ Could not save focus session: {db_error}")

        return jsonify({
            'prediction': prediction,
            'probabilities': prob_dict,
            'focus_level': focus_level,
            'features_extracted': len(features),
            'model_type': 'Stacking Classifier',
            'status': 'success'
        })

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'details': 'Check server logs'}), 500


# ===============================
# CSV Upload Prediction Endpoint
# ===============================
@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    """
    CSV file prediction endpoint.
    
    Accepts a multipart/form-data POST with:
      - file: the CSV file
      - fs  : (optional) sample rate, default 256
      - column: (optional) column name to use as signal, default auto-detect
    
    Supported CSV formats:
      1. Single column of raw EEG values (no header)
      2. Single column with header (e.g. "eeg" or "value")
      3. Multi-column — specify ?column=<name> or it auto-picks the first numeric column
    
    Returns: same structure as /predict, plus per-window breakdown.
    """
    try:
        if not MODEL_LOADED:
            return jsonify({'error': 'Model not loaded.'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Send as multipart field "file".'}), 400

        file = request.files['file']
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only .csv files are supported.'}), 400

        fs = int(request.form.get('fs', 256))
        target_column = request.form.get('column', None)  # optional column name

        # ── Parse CSV ──────────────────────────────────────────────────
        content = file.read().decode('utf-8', errors='replace')
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            return jsonify({'error': 'CSV file is empty.'}), 400

        # Detect header row
        has_header = False
        header = []
        try:
            float(rows[0][0].strip())
        except (ValueError, IndexError):
            has_header = True
            header = [h.strip() for h in rows[0]]
            rows = rows[1:]

        if not rows:
            return jsonify({'error': 'CSV has header but no data rows.'}), 400

        # Find which column index to use
        col_idx = 0
        if target_column and has_header:
            if target_column in header:
                col_idx = header.index(target_column)
            else:
                return jsonify({'error': f'Column "{target_column}" not found. Available: {header}'}), 400
        elif has_header and len(header) > 1:
            # Auto-pick first numeric column
            for i, h in enumerate(header):
                try:
                    float(rows[0][i].strip())
                    col_idx = i
                    break
                except (ValueError, IndexError):
                    continue

        # Extract signal values
        signal = []
        skipped = 0
        for row in rows:
            if not row or col_idx >= len(row):
                skipped += 1
                continue
            try:
                val = float(row[col_idx].strip())
                if np.isfinite(val):
                    signal.append(val)
                else:
                    skipped += 1
            except (ValueError, IndexError):
                skipped += 1

        if len(signal) < 64:
            return jsonify({
                'error': f'Not enough valid numeric samples (got {len(signal)}, need ≥ 64).',
                'skipped_rows': skipped
            }), 400

        signal = np.array(signal, dtype=np.float64)
        print(f"📂 CSV loaded: {len(signal)} samples, {skipped} skipped, fs={fs}Hz")

        # ── Slide window across signal & predict each window ───────────
        WINDOW_SEC = 4
        window_size = fs * WINDOW_SEC
        step_size = fs * 2  # 2-second step (50% overlap)

        window_results = []
        all_focus_levels = []
        all_prob_dicts = []
        state_counts = {}

        if len(signal) < window_size:
            # Signal shorter than one window — predict on whole signal
            features = extract_features_from_raw_signal(signal.tolist(), fs)
            prediction, prob_dict, focus_level = run_prediction_pipeline(features)
            window_results.append({
                'window': 1,
                'start_s': 0,
                'end_s': round(len(signal) / fs, 2),
                'prediction': prediction,
                'focus_level': round(focus_level, 2),
                'probabilities': prob_dict
            })
            all_focus_levels.append(focus_level)
            if prob_dict:
                all_prob_dicts.append(prob_dict)
            state_counts[prediction] = state_counts.get(prediction, 0) + 1
        else:
            starts = range(0, len(signal) - window_size + 1, step_size)
            for w_idx, start in enumerate(starts):
                end = start + window_size
                window = signal[start:end].tolist()

                features = extract_features_from_raw_signal(window, fs)
                prediction, prob_dict, focus_level = run_prediction_pipeline(features)

                window_results.append({
                    'window': w_idx + 1,
                    'start_s': round(start / fs, 2),
                    'end_s': round(end / fs, 2),
                    'prediction': prediction,
                    'focus_level': round(focus_level, 2),
                    'probabilities': prob_dict
                })
                all_focus_levels.append(focus_level)
                if prob_dict:
                    all_prob_dicts.append(prob_dict)
                state_counts[prediction] = state_counts.get(prediction, 0) + 1

            print(f"🔬 Processed {len(window_results)} windows")

        # ── Aggregate results ──────────────────────────────────────────
        avg_focus_level = float(np.mean(all_focus_levels))
        dominant_state = max(state_counts, key=state_counts.get)

        avg_probabilities = {}
        if all_prob_dicts:
            all_keys = set(k for d in all_prob_dicts for k in d)
            for key in all_keys:
                vals = [d.get(key, 0) for d in all_prob_dicts]
                avg_probabilities[key] = round(float(np.mean(vals)), 4)

        duration_s = len(signal) / fs

        # ── Auto-save to Brain Journal DB ──────────────────────────────
        try:
            focus_score = avg_focus_level / 100.0
            focus_label = "High" if focus_score >= 0.7 else ("Medium" if focus_score >= 0.4 else "Low")
            on_focus_prediction_complete(duration_s, focus_score, focus_label)
            print(f"💾 CSV session saved to Journal DB (label={focus_label})")
        except Exception as db_error:
            print(f"⚠️ Could not save to Journal DB: {db_error}")

        return jsonify({
            'status': 'success',
            'source': 'csv',
            'filename': file.filename,
            'total_samples': len(signal),
            'duration_s': round(duration_s, 2),
            'sample_rate': fs,
            'windows_processed': len(window_results),
            'skipped_rows': skipped,
            # Summary (matches /predict response shape)
            'prediction': dominant_state,
            'focus_level': round(avg_focus_level, 2),
            'probabilities': avg_probabilities,
            'model_type': 'Stacking Classifier',
            # Detail
            'state_counts': state_counts,
            'window_results': window_results,
        })

    except Exception as e:
        print(f"❌ CSV prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'details': 'Check server logs'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': str(np.datetime64('now'))
    })

def on_focus_prediction_complete(duration_s, score, label):
    try:
        requests.post("http://localhost:5000/api/sessions/focus", json={
            "duration_s":  duration_s,
            "focus_score": score,
            "focus_label": label,
            "notes":       ""
        }, timeout=2)
    except Exception:
        pass  # Journal DB is optional — don't crash the main app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 EEG Classification API Server - Stacking Model")
    print("="*60)
    print(f"Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Loaded'}")
    if MODEL_LOADED:
        print(f"Model Type: Stacking Classifier")
        print(f"Expected Features: 43 → 20 (after preprocessing)")
    print("="*60)
    print("Starting server on http://127.0.0.1:8001")
    print("\nEndpoints:")
    print("  GET  /test        - Test if API is running")
    print("  POST /predict     - Live serial prediction")
    print("  POST /predict-csv - CSV file prediction")
    print("  GET  /health      - Health check")
    print("="*60 + "\n")

    port = int(os.environ.get('PORT', 8001))
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)