import serial 
import csv 
import numpy as np 
from datetime import datetime 
from scipy.signal import welch 
from scipy.stats import skew, kurtosis
import time

# -------- SETTINGS -------- 
PORT = "COM6"
BAUD = 115200 
SAMPLE_RATE = 250   # Hz
WINDOW_SEC = 2
# -------------------------- 

# Emotion labels - Core emotions for Mood Recognition
EMOTIONS = {
    '1': 'happiness',
    '2': 'anger',
    '3': 'sadness',
    '4': 'fear',
    '5': 'neutral'
}

def compute_comprehensive_features(eeg_window, sample_rate=250):
    """
    Compute ALL features needed for emotion recognition ML models
    """
    features = {}
    
    # ===== TIME DOMAIN FEATURES =====
    features['mean'] = np.mean(eeg_window)
    features['std'] = np.std(eeg_window)
    features['variance'] = np.var(eeg_window)
    features['peak_to_peak'] = np.ptp(eeg_window)
    features['rms'] = np.sqrt(np.mean(eeg_window**2))  # Root Mean Square
    features['skewness'] = skew(eeg_window)  # Asymmetry of distribution
    features['kurtosis'] = kurtosis(eeg_window)  # Tailedness of distribution
    
    # Zero crossing rate (signal complexity indicator)
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(eeg_window)) != 0) / len(eeg_window)
    
    # Signal energy
    features['signal_energy'] = np.sum(eeg_window**2)
    
    # First and second derivatives (rate of change)
    first_diff = np.diff(eeg_window)
    features['first_diff_mean'] = np.mean(first_diff)
    features['first_diff_std'] = np.std(first_diff)
    
    second_diff = np.diff(first_diff)
    features['second_diff_mean'] = np.mean(second_diff)
    features['second_diff_std'] = np.std(second_diff)
    
    # ===== FREQUENCY DOMAIN FEATURES =====
    freqs, psd = welch(eeg_window, fs=sample_rate, nperseg=min(256, len(eeg_window)))
    
    # Define frequency bands (standard clinical bands)
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs < 45)
    
    # Low/Mid/High sub-bands for better emotion discrimination
    low_alpha_mask = (freqs >= 8) & (freqs < 10)
    high_alpha_mask = (freqs >= 10) & (freqs < 13)
    low_beta_mask = (freqs >= 13) & (freqs < 20)
    high_beta_mask = (freqs >= 20) & (freqs < 30)
    
    # Absolute band powers
    features['delta_power'] = np.trapezoid(psd[delta_mask], freqs[delta_mask]) if np.any(delta_mask) else 0
    features['theta_power'] = np.trapezoid(psd[theta_mask], freqs[theta_mask]) if np.any(theta_mask) else 0
    features['alpha_power'] = np.trapezoid(psd[alpha_mask], freqs[alpha_mask]) if np.any(alpha_mask) else 0
    features['beta_power'] = np.trapezoid(psd[beta_mask], freqs[beta_mask]) if np.any(beta_mask) else 0
    features['gamma_power'] = np.trapezoid(psd[gamma_mask], freqs[gamma_mask]) if np.any(gamma_mask) else 0
    
    # Sub-band powers
    features['low_alpha_power'] = np.trapezoid(psd[low_alpha_mask], freqs[low_alpha_mask]) if np.any(low_alpha_mask) else 0
    features['high_alpha_power'] = np.trapezoid(psd[high_alpha_mask], freqs[high_alpha_mask]) if np.any(high_alpha_mask) else 0
    features['low_beta_power'] = np.trapezoid(psd[low_beta_mask], freqs[low_beta_mask]) if np.any(low_beta_mask) else 0
    features['high_beta_power'] = np.trapezoid(psd[high_beta_mask], freqs[high_beta_mask]) if np.any(high_beta_mask) else 0
    
    # Total power
    total_power = np.trapezoid(psd, freqs)
    features['total_power'] = total_power
    
    # Relative band powers (normalized by total power)
    features['delta_relative'] = features['delta_power'] / (total_power + 1e-10)
    features['theta_relative'] = features['theta_power'] / (total_power + 1e-10)
    features['alpha_relative'] = features['alpha_power'] / (total_power + 1e-10)
    features['beta_relative'] = features['beta_power'] / (total_power + 1e-10)
    features['gamma_relative'] = features['gamma_power'] / (total_power + 1e-10)
    
    # ===== BAND POWER RATIOS (Critical for emotion recognition) =====
    features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)
    features['alpha_beta_ratio'] = features['alpha_power'] / (features['beta_power'] + 1e-10)
    features['alpha_theta_ratio'] = features['alpha_power'] / (features['theta_power'] + 1e-10)
    features['theta_alpha_ratio'] = features['theta_power'] / (features['alpha_power'] + 1e-10)
    
    # Engagement index (beta/alpha+theta)
    features['engagement_index'] = features['beta_power'] / (features['alpha_power'] + features['theta_power'] + 1e-10)
    
    # Arousal index (beta+gamma/alpha+theta)
    features['arousal_index'] = (features['beta_power'] + features['gamma_power']) / (features['alpha_power'] + features['theta_power'] + 1e-10)
    
    # Valence proxy (alpha asymmetry substitute - single channel approximation)
    features['valence_proxy'] = features['high_alpha_power'] / (features['low_alpha_power'] + 1e-10)
    
    # Cognitive load indicator
    features['cognitive_load'] = (features['theta_power'] + features['low_beta_power']) / (features['alpha_power'] + 1e-10)
    
    # Relaxation index
    features['relaxation_index'] = features['alpha_power'] / (features['beta_power'] + features['gamma_power'] + 1e-10)
    
    # ===== SPECTRAL FEATURES =====
    # Spectral centroid (center of mass of spectrum)
    features['spectral_centroid'] = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    
    # Spectral entropy (measure of disorder/complexity)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    # Spectral edge frequency (95% of power contained below this freq)
    cumsum_psd = np.cumsum(psd)
    features['spectral_edge_95'] = freqs[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]] if len(np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]) > 0 else 0
    
    # Peak frequency in each band
    if np.any(delta_mask) and np.sum(psd[delta_mask]) > 0:
        features['delta_peak_freq'] = freqs[delta_mask][np.argmax(psd[delta_mask])]
    else:
        features['delta_peak_freq'] = 0
        
    if np.any(theta_mask) and np.sum(psd[theta_mask]) > 0:
        features['theta_peak_freq'] = freqs[theta_mask][np.argmax(psd[theta_mask])]
    else:
        features['theta_peak_freq'] = 0
        
    if np.any(alpha_mask) and np.sum(psd[alpha_mask]) > 0:
        features['alpha_peak_freq'] = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
    else:
        features['alpha_peak_freq'] = 0
        
    if np.any(beta_mask) and np.sum(psd[beta_mask]) > 0:
        features['beta_peak_freq'] = freqs[beta_mask][np.argmax(psd[beta_mask])]
    else:
        features['beta_peak_freq'] = 0
    
    # ===== HJORTH PARAMETERS (Important for EEG analysis) =====
    # Activity (variance)
    features['hjorth_activity'] = np.var(eeg_window)
    
    # Mobility (mean frequency)
    diff_var = np.var(first_diff) if len(first_diff) > 0 else 0
    features['hjorth_mobility'] = np.sqrt(diff_var / (features['hjorth_activity'] + 1e-10))
    
    # Complexity (change in frequency)
    if len(second_diff) > 0:
        second_diff_var = np.var(second_diff)
        hjorth_mobility_diff = np.sqrt(second_diff_var / (diff_var + 1e-10))
        features['hjorth_complexity'] = hjorth_mobility_diff / (features['hjorth_mobility'] + 1e-10)
    else:
        features['hjorth_complexity'] = 0
    
    return features

def get_feature_headers():
    """Return ordered list of all feature names for CSV header"""
    return [
        # Metadata
        'timestamp_ms', 'eeg_raw_value', 'emotion_label', 'participant_id', 'session_number',
        
        # Time domain features
        'mean', 'std', 'variance', 'peak_to_peak', 'rms', 'skewness', 'kurtosis',
        'zero_crossing_rate', 'signal_energy',
        'first_diff_mean', 'first_diff_std', 'second_diff_mean', 'second_diff_std',
        
        # Absolute band powers
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
        'low_alpha_power', 'high_alpha_power', 'low_beta_power', 'high_beta_power',
        'total_power',
        
        # Relative band powers
        'delta_relative', 'theta_relative', 'alpha_relative', 'beta_relative', 'gamma_relative',
        
        # Band ratios and indices
        'theta_beta_ratio', 'alpha_beta_ratio', 'alpha_theta_ratio', 'theta_alpha_ratio',
        'engagement_index', 'arousal_index', 'valence_proxy', 'cognitive_load', 'relaxation_index',
        
        # Spectral features
        'spectral_centroid', 'spectral_entropy', 'spectral_edge_95',
        'delta_peak_freq', 'theta_peak_freq', 'alpha_peak_freq', 'beta_peak_freq',
        
        # Hjorth parameters
        'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
    ]

# ===== MAIN RECORDING SCRIPT =====

# Get participant info
participant = input("Enter participant ID (e.g., P001): ").strip()
session_num = input("Enter session number (e.g., 1, 2, 3): ").strip()

print("\n=== EMOTION RECORDING PROTOCOL ===")
print("Available emotions:")
for key, emotion in EMOTIONS.items():
    print(f"  {key}: {emotion}")

current_emotion = input("\nEnter emotion number (or type emotion name): ").strip()
if current_emotion in EMOTIONS:
    emotion_label = EMOTIONS[current_emotion]
else:
    emotion_label = current_emotion.lower()

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
FILENAME = f"{participant}_S{session_num}_{emotion_label}_{timestamp_str}.csv"

print(f"\n📝 Participant: {participant}")
print(f"📝 Session: {session_num}")
print(f"📝 Emotion: {emotion_label}")
print(f"💾 Saving to: {FILENAME}")
print("\n🎬 INSTRUCTIONS:")
if emotion_label == 'happiness':
    print("   😊 Watch funny videos or comedy clips")
    print("   😊 Listen to upbeat, joyful music")
    print("   😊 Think of your happiest memories")
    print("   😊 Smile and feel genuine joy!")
elif emotion_label == 'anger':
    print("   😠 Think of something that really frustrates you")
    print("   😠 Recall unfair situations or injustices")
    print("   😠 Remember arguments or conflicts")
    print("   😠 Feel the tension and frustration (safely!)")
elif emotion_label == 'sadness':
    print("   😢 Watch a sad/emotional movie scene")
    print("   😢 Listen to melancholic music")
    print("   😢 Think of losses or disappointments")
    print("   😢 Allow yourself to feel the sadness")
elif emotion_label == 'fear':
    print("   😨 Watch horror/thriller movie clips")
    print("   😨 Think of scary or threatening situations")
    print("   😨 Imagine things that make you anxious")
    print("   😨 Feel the nervousness and apprehension")
elif emotion_label == 'neutral':
    print("   😐 Sit comfortably with eyes open")
    print("   😐 Don't think of anything emotional")
    print("   😐 Just observe your surroundings calmly")
    print("   😐 Keep a blank, neutral expression")

print("\n⏱  Recommended duration: 2-3 minutes")
input("\nPress ENTER when ready to start recording...")

# Connect to EEG device
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"\n✓ Connected to {PORT}")
    time.sleep(2)  # Wait for connection to stabilize
except Exception as e:
    print(f"❌ Error connecting to {PORT}: {e}")
    exit(1)

print("🔴 Recording started... (Press Ctrl+C to stop)\n")

# Open CSV for writing
with open(FILENAME, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Write header
    headers = get_feature_headers()
    writer.writerow(headers)
    
    buffer = []
    sample_count = 0
    rows_written = 0
    
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or "," not in line:
                continue
            
            parts = line.split(",")
            if len(parts) != 2:
                continue
                
            t_ms_str, eeg_str = parts
            try:
                t_ms = int(t_ms_str)
                eeg_val = float(eeg_str)
            except ValueError:
                continue
            
            buffer.append(eeg_val)
            sample_count += 1
            
            # Print progress every 2 seconds
            if sample_count % (SAMPLE_RATE * 2) == 0:
                elapsed = sample_count / SAMPLE_RATE
                print(f"⏱  {elapsed:.0f}s | Samples: {sample_count} | Features computed: {rows_written}")
            
            # Compute features when window is full
            if len(buffer) >= SAMPLE_RATE * WINDOW_SEC:
                eeg_window = np.array(buffer[-SAMPLE_RATE * WINDOW_SEC:])
                
                # Compute all features
                features = compute_comprehensive_features(eeg_window, SAMPLE_RATE)
                
                # Prepare row with metadata + features
                row = [
                    t_ms, eeg_val, emotion_label, participant, session_num
                ]
                
                # Add all features in order
                for header in headers[5:]:  # Skip first 5 metadata columns
                    row.append(features.get(header, 0))
                
                writer.writerow(row)
                rows_written += 1
                
    except KeyboardInterrupt:
        print(f"\n\n✅ Recording completed successfully!")
        print(f"📊 Total samples collected: {sample_count}")
        print(f"⏱  Duration: {sample_count / SAMPLE_RATE:.1f} seconds")
        print(f"📝 Feature rows written: {rows_written}")
        print(f"💾 Data saved to: {FILENAME}")
        print(f"\n📈 Total features per row: {len(headers)}")
        ser.close()
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        ser.close()