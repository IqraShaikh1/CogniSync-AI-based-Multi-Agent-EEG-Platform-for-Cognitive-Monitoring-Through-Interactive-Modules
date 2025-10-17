# app.py - Flask API for EEG Classification
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from scipy import stats
import warnings
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
    
    # Load the stacking model
    model = joblib.load("best_model_stacking_H.pkl")
    print("✅ Loaded: best_model_stacking_H.pkl")
    
    # Load preprocessing objects
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
    
    # Basic band powers
    alpha = band_powers.get('alpha', 0)
    beta = band_powers.get('beta', 0)
    theta = band_powers.get('theta', 0)
    delta = band_powers.get('delta', 0)
    
    # Ratios
    features['theta_beta_ratio'] = theta / (beta + 1e-6)
    features['alpha_theta_ratio'] = alpha / (theta + 1e-6)
    features['alpha_plus_theta_beta'] = (alpha + theta) / (beta + 1e-6)
    features['alpha_beta_ratio_mean'] = alpha / (beta + 1e-6)
    
    # Power statistics (approximated)
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
    
    # Additional ratio features
    features['alpha_beta_ratio_std'] = 0.0
    features['alpha_beta_ratio_min'] = features['alpha_beta_ratio_mean'] * 0.8
    features['alpha_beta_ratio_max'] = features['alpha_beta_ratio_mean'] * 1.2
    features['alpha_beta_ratio_skew'] = 0.0
    features['alpha_beta_ratio_kurt'] = 0.0
    
    # Signal variance features (estimated)
    total_power = alpha + beta + theta + delta
    features['signal_variance_mean'] = total_power
    features['signal_variance_std'] = total_power * 0.1
    features['signal_variance_min'] = total_power * 0.5
    features['signal_variance_max'] = total_power * 1.5
    features['signal_variance_skew'] = 0.0
    features['signal_variance_kurt'] = 0.0
    
    # Attention index
    attention_idx = beta / (theta + alpha + 1e-6)
    features['attention_index_mean'] = attention_idx
    features['attention_index_std'] = attention_idx * 0.1
    features['attention_index_min'] = attention_idx * 0.8
    features['attention_index_max'] = attention_idx * 1.2
    features['attention_index_skew'] = 0.0
    features['attention_index_kurt'] = 0.0
    
    # Entropy features (estimated)
    features['sample_entropy'] = 0.5
    features['perm_entropy'] = 0.5
    
    # Hjorth parameters (estimated)
    features['hjorth_activity'] = total_power
    features['hjorth_mobility'] = 1.0
    features['hjorth_complexity'] = 1.0
    
    return features

def extract_features_from_raw_signal(signal, fs=256):
    """
    Extract all 43 features from raw EEG signal
    This provides better accuracy than approximated features
    """
    signal = np.array(signal)
    
    # Compute FFT for band powers
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(n, 1/fs)
    psd = np.abs(fft_vals) ** 2
    
    # Define frequency bands
    def band_power(lo, hi):
        idx = np.where((fft_freq >= lo) & (fft_freq < hi))
        return np.sum(psd[idx])
    
    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 12)
    beta = band_power(12, 30)
    
    features = {}
    
    # Band power statistics
    for band_name, band_val in [('alpha', alpha), ('beta', beta), ('theta', theta)]:
        features[f'{band_name}_power_mean'] = band_val
        features[f'{band_name}_power_std'] = band_val * 0.15
        features[f'{band_name}_power_min'] = band_val * 0.5
        features[f'{band_name}_power_max'] = band_val * 1.5
        features[f'{band_name}_power_skew'] = 0.0
        features[f'{band_name}_power_kurt'] = 0.0
    
    # Ratios
    features['theta_beta_ratio'] = theta / (beta + 1e-6)
    features['alpha_theta_ratio'] = alpha / (theta + 1e-6)
    features['alpha_plus_theta_beta'] = (alpha + theta) / (beta + 1e-6)
    features['alpha_beta_ratio_mean'] = alpha / (beta + 1e-6)
    features['alpha_beta_ratio_std'] = 0.0
    features['alpha_beta_ratio_min'] = features['alpha_beta_ratio_mean'] * 0.8
    features['alpha_beta_ratio_max'] = features['alpha_beta_ratio_mean'] * 1.2
    features['alpha_beta_ratio_skew'] = 0.0
    features['alpha_beta_ratio_kurt'] = 0.0
    
    # Signal variance
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
    
    # Attention index
    attention_idx = beta / (theta + alpha + 1e-6)
    features['attention_index_mean'] = attention_idx
    features['attention_index_std'] = attention_idx * 0.1
    features['attention_index_min'] = attention_idx * 0.8
    features['attention_index_max'] = attention_idx * 1.2
    features['attention_index_skew'] = 0.0
    features['attention_index_kurt'] = 0.0
    
    # Entropy (simplified)
    features['sample_entropy'] = 0.5
    features['perm_entropy'] = 0.5
    
    # Hjorth parameters
    features['hjorth_activity'] = np.var(signal)
    diff1 = np.diff(signal)
    features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(signal) + 1e-6))
    diff2 = np.diff(diff1)
    features['hjorth_complexity'] = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-6)) / (features['hjorth_mobility'] + 1e-6)
    
    return features

# ===============================
# API Endpoints
# ===============================
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify API is running"""
    return jsonify({
        'status': 'success',
        'message': 'EEG Classification API is running',
        'model_loaded': MODEL_LOADED,
        'model_type': 'Stacking Classifier'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts either:
    1. Band powers: {'delta': float, 'theta': float, 'alpha': float, 'beta': float}
    2. Raw signal: {'signal': [float, ...], 'fs': int}
    """
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded. Please run save_preprocessing.py first!'
            }), 500
        
        data = request.json
        
        # Check if raw signal is provided
        if 'signal' in data:
            signal = data['signal']
            fs = data.get('fs', 256)
            features = extract_features_from_raw_signal(signal, fs)
            print(f"📊 Extracted features from {len(signal)} samples at {fs}Hz")
        else:
            # Extract features from band powers
            features = compute_band_power_features(data)
            print(f"📊 Extracted features from band powers")
        
        # Create feature vector in correct order (43 features)
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
        
        X = np.array([[features.get(name, 0.0) for name in feature_names]])
        
        # Apply preprocessing pipeline
        # 1. Variance threshold
        X_var = variance_selector.transform(X)
        
        # 2. Remove correlated features (use saved column names)
        import pandas as pd
        kept_features = [name for name, keep in zip(feature_names, variance_selector.get_support()) if keep]
        X_df = pd.DataFrame(X_var, columns=kept_features)
        X_sel = X_df[selected_features]
        
        # 3. Scaling
        X_scaled = scaler.transform(X_sel)
        
        # 4. Feature selection
        X_final = feature_selector.transform(X_scaled)
        
        print(f"🔬 Feature pipeline: 43 → {X_var.shape[1]} → {X_sel.shape[1]} → {X_final.shape[1]}")
        
        # Make prediction with stacking model
        prediction = model.predict(X_final)[0]
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(X_final)[0]
            class_names = model.classes_
            prob_dict = {str(cls): float(prob) for cls, prob in zip(class_names, probabilities)}
            print(f"🎯 Prediction: {prediction} (confidence: {max(probabilities):.2%})")
        except Exception as e:
            print(f"⚠️ Could not get probabilities: {e}")
            prob_dict = None
        
        # Compute focus level based on model prediction and confidence
        # This is more accurate than a simple formula
        if prob_dict:
            # Focus states: 1 and 2
            focus_prob = prob_dict.get('1', 0) + prob_dict.get('2', 0)
            
            # Distracted state: 3
            distracted_prob = prob_dict.get('3', 0)
            
            # Baseline: 0
            baseline_prob = prob_dict.get('0', 0)
            
            # Calculate focus level based on probabilities
            # Focus states contribute positively, distracted negatively
            focus_level = (focus_prob * 100) - (distracted_prob * 20)
            focus_level = max(0, min(100, focus_level))
            
            print(f"📊 Model-based Focus Level: {focus_level:.1f}%")
            print(f"   Focus1+Focus2: {focus_prob:.1%}, Distracted: {distracted_prob:.1%}, Baseline: {baseline_prob:.1%}")
        else:
            # Fallback to band power calculation if probabilities unavailable
            if 'signal' in data:
                alpha = features.get('alpha_power_mean', 0)
                beta = features.get('beta_power_mean', 0)
                theta = features.get('theta_power_mean', 0)
                delta = features.get('delta', 0)
            else:
                alpha = data.get('alpha', 0)
                beta = data.get('beta', 0)
                theta = data.get('theta', 0)
                delta = data.get('delta', 0)
            
            # Simple engagement ratio
            focus_level = (beta / (theta + alpha + 1e-6)) * 50
            focus_level = max(0, min(100, focus_level))
        
        return jsonify({
            'prediction': str(prediction),
            'probabilities': prob_dict,
            'focus_level': float(focus_level),
            'features_extracted': len(features),
            'model_type': 'Stacking Classifier',
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

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 EEG Classification API Server - Stacking Model")
    print("="*60)
    print(f"Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Loaded'}")
    if MODEL_LOADED:
        print(f"Model Type: Stacking Classifier")
        print(f"Expected Features: 43 → 20 (after preprocessing)")
    print("="*60)
    print("Starting server on http://127.0.0.1:8000")
    print("\nEndpoints:")
    print("  GET  /test    - Test if API is running")
    print("  POST /predict - Get EEG classification")
    print("  GET  /health  - Health check")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=8000, debug=True)
    