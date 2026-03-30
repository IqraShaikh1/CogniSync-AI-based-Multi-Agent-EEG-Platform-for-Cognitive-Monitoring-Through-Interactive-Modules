# app_emotion.py - Flask API for EEG Emotion Classification
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from scipy import stats, signal
import warnings
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

# Emotion label mapping (string to number)
EMOTION_TO_NUM = {
    'happiness': 1,
    'anger': 2,
    'sadness': 3,
    'fear': 4,
    'neutral': 5
}

# Reverse mapping (number to string)
NUM_TO_EMOTION = {
    1: "Happiness",
    2: "Anger",
    3: "Sadness",
    4: "Fear",
    5: "Neutral"
}

# ===============================
# Load Model and Preprocessing Objects
# ===============================
MODEL_LOADED = False
model = None
scaler = None
label_encoder = None

try:
    print("Loading emotion model and preprocessing objects...")
    
    # Load the emotion model
    model = joblib.load("eeg_emotion_xgb_model.pkl")
    print("✅ Loaded: eeg_emotion_xgb_model.pkl")
    
    # Load preprocessing objects
    scaler = joblib.load("eeg_scaler_emotion2.pkl")
    print("✅ Loaded: eeg_scaler_emotion2.pkl")
    
    label_encoder = joblib.load("eeg_label_encoder-EMOTION2.pkl")
    print("✅ Loaded: eeg_label_encoder-EMOTION2.pkl")
    
    MODEL_LOADED = True
    print("\n✅ All emotion components loaded successfully!")
    print(f"Loaded classes: {label_encoder.classes_}")
    print("\n")
    
except Exception as e:
    print(f"\n❌ Error loading model/preprocessing objects: {e}")
    print("Please ensure these files exist:")
    print("  - eeg_emotion_xgb_model.pkl")
    print("  - eeg_scaler_emotion2.pkl")
    print("  - eeg_label_encoder-EMOTION2.pkl\n")
    MODEL_LOADED = False

# ===============================
# Feature Extraction Functions
# ===============================
def compute_statistical_features(signal_data):
    """Compute basic statistical features from raw signal"""
    signal_array = np.array(signal_data)
    
    features = {
        'mean': np.mean(signal_array),
        'std': np.std(signal_array),
        'variance': np.var(signal_array),
        'peak_to_peak': np.ptp(signal_array),
        'rms': np.sqrt(np.mean(signal_array**2)),
        'skewness': stats.skew(signal_array),
        'kurtosis': stats.kurtosis(signal_array),
        'zero_crossing_rate': np.sum(np.diff(np.signbit(signal_array))) / len(signal_array),
        'signal_energy': np.sum(signal_array**2)
    }
    
    # First and second derivatives
    first_diff = np.diff(signal_array)
    second_diff = np.diff(first_diff)
    
    features['first_diff_mean'] = np.mean(first_diff) if len(first_diff) > 0 else 0
    features['first_diff_std'] = np.std(first_diff) if len(first_diff) > 0 else 0
    features['second_diff_mean'] = np.mean(second_diff) if len(second_diff) > 0 else 0
    features['second_diff_std'] = np.std(second_diff) if len(second_diff) > 0 else 0
    
    return features

def compute_band_powers(signal_data, fs=256):
    """Compute frequency band powers using FFT"""
    signal_array = np.array(signal_data)
    n = len(signal_array)
    
    # Apply Hanning window
    windowed = signal_array * np.hanning(n)
    
    # Compute FFT
    fft_vals = np.fft.rfft(windowed)
    fft_freq = np.fft.rfftfreq(n, 1/fs)
    psd = np.abs(fft_vals) ** 2
    
    # Define frequency bands
    def band_power(lo, hi):
        idx = np.where((fft_freq >= lo) & (fft_freq < hi))[0]
        return np.sum(psd[idx]) if len(idx) > 0 else 0
    
    # Compute band powers
    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 12)
    beta = band_power(12, 30)
    gamma = band_power(30, 50)
    
    # Sub-bands
    low_alpha = band_power(8, 10)
    high_alpha = band_power(10, 12)
    low_beta = band_power(12, 20)
    high_beta = band_power(20, 30)
    
    total_power = delta + theta + alpha + beta + gamma
    
    band_features = {
        'delta_power': delta,
        'theta_power': theta,
        'alpha_power': alpha,
        'beta_power': beta,
        'gamma_power': gamma,
        'low_alpha_power': low_alpha,
        'high_alpha_power': high_alpha,
        'low_beta_power': low_beta,
        'high_beta_power': high_beta,
        'total_power': total_power
    }
    
    # Relative powers
    if total_power > 0:
        band_features['delta_relative'] = delta / total_power
        band_features['theta_relative'] = theta / total_power
        band_features['alpha_relative'] = alpha / total_power
        band_features['beta_relative'] = beta / total_power
        band_features['gamma_relative'] = gamma / total_power
    else:
        band_features['delta_relative'] = 0
        band_features['theta_relative'] = 0
        band_features['alpha_relative'] = 0
        band_features['beta_relative'] = 0
        band_features['gamma_relative'] = 0
    
    # Band ratios
    band_features['theta_beta_ratio'] = theta / (beta + 1e-10)
    band_features['alpha_beta_ratio'] = alpha / (beta + 1e-10)
    band_features['alpha_theta_ratio'] = alpha / (theta + 1e-10)
    band_features['theta_alpha_ratio'] = theta / (alpha + 1e-10)
    
    # Engagement and arousal indices
    band_features['engagement_index'] = beta / (alpha + theta + 1e-10)
    band_features['arousal_index'] = (beta + gamma) / (alpha + theta + 1e-10)
    band_features['valence_proxy'] = (alpha - beta) / (alpha + beta + 1e-10)
    band_features['cognitive_load'] = theta / alpha if alpha > 0 else 0
    band_features['relaxation_index'] = alpha / beta if beta > 0 else 0
    
    # Spectral features
    if len(psd) > 0 and total_power > 0:
        # Spectral centroid
        band_features['spectral_centroid'] = np.sum(fft_freq * psd) / np.sum(psd)
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        band_features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Spectral edge frequency (95%)
        cumsum_psd = np.cumsum(psd)
        idx_95 = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
        band_features['spectral_edge_95'] = fft_freq[idx_95[0]] if len(idx_95) > 0 else 0
    else:
        band_features['spectral_centroid'] = 0
        band_features['spectral_entropy'] = 0
        band_features['spectral_edge_95'] = 0
    
    # Peak frequencies in each band
    def peak_frequency(lo, hi):
        idx = np.where((fft_freq >= lo) & (fft_freq < hi))[0]
        if len(idx) > 0 and len(psd[idx]) > 0:
            peak_idx = idx[np.argmax(psd[idx])]
            return fft_freq[peak_idx]
        return 0
    
    band_features['delta_peak_freq'] = peak_frequency(0.5, 4)
    band_features['theta_peak_freq'] = peak_frequency(4, 8)
    band_features['alpha_peak_freq'] = peak_frequency(8, 12)
    band_features['beta_peak_freq'] = peak_frequency(12, 30)
    
    return band_features

def compute_hjorth_parameters(signal_data):
    """Compute Hjorth parameters (Activity, Mobility, Complexity)"""
    signal_array = np.array(signal_data)
    
    # Activity
    activity = np.var(signal_array)
    
    # Mobility
    first_diff = np.diff(signal_array)
    mobility = np.sqrt(np.var(first_diff) / (activity + 1e-10))
    
    # Complexity
    second_diff = np.diff(first_diff)
    complexity = np.sqrt(np.var(second_diff) / (np.var(first_diff) + 1e-10)) / (mobility + 1e-10)
    
    return {
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    }

def extract_all_features(signal_data, fs=256):
    """
    Extract all features from raw EEG signal
    Returns feature vector matching the training data format
    """
    features = {}
    
    # Statistical features
    stat_features = compute_statistical_features(signal_data)
    features.update(stat_features)
    
    # Band power features
    band_features = compute_band_powers(signal_data, fs)
    features.update(band_features)
    
    # Hjorth parameters
    hjorth_features = compute_hjorth_parameters(signal_data)
    features.update(hjorth_features)
    
    return features

# ===============================
# API Endpoints
# ===============================
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify API is running"""
    emotions_list = []
    if MODEL_LOADED and label_encoder:
        for class_label in label_encoder.classes_:
            if isinstance(class_label, str):
                emotion_num = EMOTION_TO_NUM.get(class_label.lower(), 0)
                emotion_name = NUM_TO_EMOTION.get(emotion_num, class_label)
            else:
                emotion_name = NUM_TO_EMOTION.get(class_label, f"Emotion {class_label}")
            emotions_list.append(emotion_name)
    
    return jsonify({
        'status': 'success',
        'message': 'EEG Emotion Classification API is running',
        'model_loaded': MODEL_LOADED,
        'model_type': 'XGBoost Classifier',
        'emotions': emotions_list
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts raw signal: {'signal': [float, ...], 'fs': int}
    Or band powers: {'delta': float, 'theta': float, 'alpha': float, 'beta': float, 'gamma': float}
    """
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist!'
            }), 500
        
        data = request.json
        
        # Check if raw signal is provided
        if 'signal' in data:
            signal_data = data['signal']
            fs = data.get('fs', 256)
            
            if len(signal_data) < 64:
                return jsonify({
                    'error': 'Signal too short. Need at least 64 samples.'
                }), 400
            
            print(f"📊 Extracting features from {len(signal_data)} samples at {fs}Hz")
            features = extract_all_features(signal_data, fs)
            
        else:
            # Use provided band powers (less accurate fallback)
            print("📊 Using provided band powers")
            features = {
                'delta_power': data.get('delta', 0),
                'theta_power': data.get('theta', 0),
                'alpha_power': data.get('alpha', 0),
                'beta_power': data.get('beta', 0),
                'gamma_power': data.get('gamma', 0),
            }
            
            # Estimate other features
            total_power = sum(features.values())
            features['total_power'] = total_power
            
            if total_power > 0:
                features['delta_relative'] = features['delta_power'] / total_power
                features['theta_relative'] = features['theta_power'] / total_power
                features['alpha_relative'] = features['alpha_power'] / total_power
                features['beta_relative'] = features['beta_power'] / total_power
                features['gamma_relative'] = features['gamma_power'] / total_power
            
            # Add minimal required features with default values
            default_features = {
                'mean': 0, 'std': 0, 'variance': 0, 'peak_to_peak': 0, 'rms': 0,
                'skewness': 0, 'kurtosis': 0, 'zero_crossing_rate': 0, 'signal_energy': 0,
                'first_diff_mean': 0, 'first_diff_std': 0, 'second_diff_mean': 0,
                'second_diff_std': 0, 'low_alpha_power': 0, 'high_alpha_power': 0,
                'low_beta_power': 0, 'high_beta_power': 0, 'alpha_theta_ratio': 0,
                'theta_alpha_ratio': 0, 'engagement_index': 0, 'cognitive_load': 0,
                'relaxation_index': 0, 'spectral_centroid': 0, 'spectral_entropy': 0,
                'spectral_edge_95': 0, 'delta_peak_freq': 0, 'theta_peak_freq': 0,
                'alpha_peak_freq': 0, 'beta_peak_freq': 0, 'hjorth_activity': 0,
                'hjorth_mobility': 0, 'hjorth_complexity': 0
            }
            
            # Add calculated ratios
            features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)
            features['alpha_beta_ratio'] = features['alpha_power'] / (features['beta_power'] + 1e-10)
            features['arousal_index'] = (features['beta_power'] + features['gamma_power']) / (features['alpha_power'] + features['theta_power'] + 1e-10)
            features['valence_proxy'] = (features['alpha_power'] - features['beta_power']) / (features['alpha_power'] + features['beta_power'] + 1e-10)
            
            # Merge with defaults
            features = {**default_features, **features}
        
        # Expected feature order (based on training CSV columns)
        feature_names = [
            'mean', 'std', 'variance', 'peak_to_peak', 'rms', 'skewness', 'kurtosis',
            'zero_crossing_rate', 'signal_energy', 'first_diff_mean', 'first_diff_std',
            'second_diff_mean', 'second_diff_std', 'delta_power', 'theta_power',
            'alpha_power', 'beta_power', 'gamma_power', 'low_alpha_power',
            'high_alpha_power', 'low_beta_power', 'high_beta_power', 'total_power',
            'delta_relative', 'theta_relative', 'alpha_relative', 'beta_relative',
            'gamma_relative', 'theta_beta_ratio', 'alpha_beta_ratio', 'alpha_theta_ratio',
            'theta_alpha_ratio', 'engagement_index', 'arousal_index', 'valence_proxy',
            'cognitive_load', 'relaxation_index', 'spectral_centroid', 'spectral_entropy',
            'spectral_edge_95', 'delta_peak_freq', 'theta_peak_freq', 'alpha_peak_freq',
            'beta_peak_freq', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
        ]
        
        # Create feature vector
        X = np.array([[features.get(name, 0.0) for name in feature_names]])
        
        print(f"🔬 Feature vector shape: {X.shape}")
        print(f"📊 Sample features: arousal={features.get('arousal_index', 0):.3f}, valence={features.get('valence_proxy', 0):.3f}")
        
        # Apply preprocessing
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction_encoded = model.predict(X_scaled)[0]
        
        # Convert prediction to emotion number
        if isinstance(prediction_encoded, str):
            # If model returns string like 'anger', 'happiness', etc.
            emotion_num = EMOTION_TO_NUM.get(prediction_encoded.lower(), 5)
            emotion_name = NUM_TO_EMOTION.get(emotion_num, prediction_encoded)
        else:
            # If model returns number
            emotion_num = int(prediction_encoded)
            emotion_name = NUM_TO_EMOTION.get(emotion_num, f"Emotion {emotion_num}")
        
        # Get probabilities
        try:
            probabilities_raw = model.predict_proba(X_scaled)[0]
            
            # Map probabilities to emotion numbers
            prob_dict = {}
            for i, prob in enumerate(probabilities_raw):
                class_label = label_encoder.classes_[i]
                if isinstance(class_label, str):
                    num = EMOTION_TO_NUM.get(class_label.lower(), i+1)
                else:
                    num = int(class_label)
                prob_dict[str(num)] = float(prob)
            
            confidence = float(max(probabilities_raw))
            print(f"🎯 Prediction: {emotion_name} ({emotion_num}) - confidence: {confidence:.2%}")
        except Exception as e:
            print(f"⚠️ Could not get probabilities: {e}")
            prob_dict = {str(emotion_num): 1.0}
            confidence = 1.0
        
        # Compute arousal and valence indices
        arousal_index = features.get('arousal_index', 0)
        valence_proxy = features.get('valence_proxy', 0)
        
        # Normalize to 0-100 scale with proper clamping
        # Arousal: typically ranges from 0-5, so divide by 5 and multiply by 100
        arousal_scaled = min(100, max(0, (arousal_index / 5.0) * 100))
        # Valence: ranges from -1 to +1, so normalize to 0-100
        valence_scaled = min(100, max(0, ((valence_proxy + 1) / 2.0) * 100))
        
        print(f"📊 Raw arousal: {arousal_index:.3f}, scaled: {arousal_scaled:.1f}")
        print(f"📊 Raw valence: {valence_proxy:.3f}, scaled: {valence_scaled:.1f}")
        # Auto-store to journal
        try:
            requests.post("http://127.0.0.1:5000/api/sessions/emotion", json={
            "duration_s": 4,  # or actual duration if you calculate it
            "dominant_emotion": emotion_name,
            "emotion_scores": prob_dict,
            "notes": ""
            })
            print("✅ Emotion session stored in journal DB")
        except Exception as e:
            print("⚠️ Could not store emotion session:", e)

        
        return jsonify({
            'prediction': int(emotion_num),
            'emotion': emotion_name,
            'probabilities': prob_dict,
            'confidence': confidence,
            'arousal_index': float(arousal_scaled),
            'valence_proxy': float(valence_scaled),
            'features_extracted': len(features),
            'model_type': 'XGBoost Classifier',
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'details': 'Check server logs for more information'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': str(np.datetime64('now'))
    })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Process entire CSV file of EEG data
    Accepts: {'eeg_values': [float, ...], 'fs': int}
    Returns: Aggregated predictions across the entire dataset
    """
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist!'
            }), 500
        
        data = request.json
        eeg_values = data.get('eeg_values', [])
        fs = data.get('fs', 256)
        window_size = fs * 4  # 4-second windows
        
        if len(eeg_values) < window_size:
            return jsonify({
                'error': f'Insufficient data. Need at least {window_size} samples, got {len(eeg_values)}'
            }), 400
        
        print(f"📊 Processing CSV with {len(eeg_values)} samples...")
        
        predictions = []
        step = window_size // 2  # 50% overlap
        
        # Process in windows
        window_count = 0
        for i in range(0, len(eeg_values) - window_size + 1, step):
            window = eeg_values[i:i + window_size]
            features = extract_all_features(window, fs)
            
            # Create feature vector
            feature_names = [
                'mean', 'std', 'variance', 'peak_to_peak', 'rms', 'skewness', 'kurtosis',
                'zero_crossing_rate', 'signal_energy', 'first_diff_mean', 'first_diff_std',
                'second_diff_mean', 'second_diff_std', 'delta_power', 'theta_power',
                'alpha_power', 'beta_power', 'gamma_power', 'low_alpha_power',
                'high_alpha_power', 'low_beta_power', 'high_beta_power', 'total_power',
                'delta_relative', 'theta_relative', 'alpha_relative', 'beta_relative',
                'gamma_relative', 'theta_beta_ratio', 'alpha_beta_ratio', 'alpha_theta_ratio',
                'theta_alpha_ratio', 'engagement_index', 'arousal_index', 'valence_proxy',
                'cognitive_load', 'relaxation_index', 'spectral_centroid', 'spectral_entropy',
                'spectral_edge_95', 'delta_peak_freq', 'theta_peak_freq', 'alpha_peak_freq',
                'beta_peak_freq', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
            ]
            
            X = np.array([[features.get(name, 0.0) for name in feature_names]])
            X_scaled = scaler.transform(X)
            
            prediction_encoded = model.predict(X_scaled)[0]
            
            if isinstance(prediction_encoded, str):
                emotion_num = EMOTION_TO_NUM.get(prediction_encoded.lower(), 5)
            else:
                emotion_num = int(prediction_encoded)
            
            predictions.append({
                'emotion_num': emotion_num,
                'arousal': features.get('arousal_index', 0),
                'valence': features.get('valence_proxy', 0)
            })
            
            window_count += 1
            if window_count <= 3:
                print(f"   Window {window_count}: {NUM_TO_EMOTION.get(emotion_num)} ({emotion_num})")
            elif window_count % 10 == 0:
                print(f"   Processed {window_count} windows...")
        
        # Aggregate results
        if len(predictions) == 0:
            return jsonify({'error': 'No predictions generated'}), 400
        
        # Count emotions
        emotion_counts = {}
        for pred in predictions:
            emotion_counts[pred['emotion_num']] = emotion_counts.get(pred['emotion_num'], 0) + 1
        
        print(f"📊 Emotion distribution:")
        for enum, count in sorted(emotion_counts.items()):
            percentage = (count / len(predictions)) * 100
            print(f"   {NUM_TO_EMOTION.get(enum)}: {count} ({percentage:.1f}%)")
        
        # Most common emotion
        dominant_emotion_num = max(emotion_counts, key=emotion_counts.get)
        dominant_emotion = NUM_TO_EMOTION.get(dominant_emotion_num, f"Emotion {dominant_emotion_num}")
        
        # Average arousal and valence with better scaling
        avg_arousal_raw = np.mean([p['arousal'] for p in predictions])
        avg_valence_raw = np.mean([p['valence'] for p in predictions])
        
        # Proper normalization
        avg_arousal = min(100, max(0, (avg_arousal_raw / 5.0) * 100))
        avg_valence = min(100, max(0, ((avg_valence_raw + 1) / 2.0) * 100))
        
        print(f"📊 Avg arousal raw: {avg_arousal_raw:.3f}, scaled: {avg_arousal:.1f}")
        print(f"📊 Avg valence raw: {avg_valence_raw:.3f}, scaled: {avg_valence:.1f}")
        
        # Probabilities (based on counts)
        total_predictions = len(predictions)
        prob_dict = {}
        for emotion_num in range(1, 6):
            count = emotion_counts.get(emotion_num, 0)
            prob_dict[str(emotion_num)] = count / total_predictions
        
        print(f"✅ CSV processed: {len(predictions)} windows, dominant emotion: {dominant_emotion}")
        # Auto-store aggregated session
        try:
            requests.post("http://127.0.0.1:5000/api/sessions/emotion", json={
            "duration_s": len(predictions) * 4,
            "dominant_emotion": dominant_emotion,
            "emotion_scores": prob_dict,
            "notes": "CSV batch session"
            })
            print("✅ CSV emotion session stored in journal DB")
        except Exception as e:
            print("⚠️ Could not store CSV session:", e)

        
        return jsonify({
            'prediction': int(dominant_emotion_num),
            'emotion': dominant_emotion,
            'probabilities': prob_dict,
            'arousal_index': float(min(100, max(0, avg_arousal))),
            'valence_proxy': float(min(100, max(0, avg_valence))),
            'total_windows': len(predictions),
            'emotion_counts': {NUM_TO_EMOTION.get(k, str(k)): v for k, v in emotion_counts.items()},
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in CSV prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'details': 'Check server logs for more information'
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("😊 EEG Emotion Classification API Server")
    print("="*60)
    print(f"Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Loaded'}")
    if MODEL_LOADED:
        print(f"Model Type: XGBoost Classifier")
        
        # Safe emotion printing
        emotions_display = []
        for class_label in label_encoder.classes_:
            if isinstance(class_label, str):
                emotion_num = EMOTION_TO_NUM.get(class_label.lower(), 0)
                emotion_name = NUM_TO_EMOTION.get(emotion_num, class_label)
            else:
                emotion_name = NUM_TO_EMOTION.get(class_label, f"Emotion {class_label}")
            emotions_display.append(emotion_name)
        
        print(f"Emotions: {', '.join(emotions_display)}")
        print(f"Features: 47 extracted features")
    print("="*60)
    print("Starting server on http://127.0.0.1:8001")
    print("\n⚠️  NOTE: Running on PORT 8001 (different from focus/attention API)")
    print("\nEndpoints:")
    print("  GET  /test       - Test if API is running")
    print("  POST /predict    - Get EEG emotion classification")
    print("  POST /predict_csv - Process entire CSV file")
    print("  GET  /health     - Health check")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 8001))   # 8001 = standalone fallback
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)