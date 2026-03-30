"""
Train Seizure Detection XGBoost Model
USING FOCUS MODULE FEATURE STANDARD

- Dataset: Bonn EEG
- Sampling rate normalized to 250 Hz
- Window: 2 seconds (500 samples)
- Features: EXACT focus module features
"""

import os
import numpy as np
import joblib
from scipy.signal import welch, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ==============================
# CONFIG
# ==============================
BONN_PATH = "Dataset\Bonn Univeristy Dataset"   # Folder with A, B, C, D, E subfolders
ORIG_FS = 173.61
TARGET_FS = 250
WINDOW_SEC = 2
WINDOW_SAMPLES = TARGET_FS * WINDOW_SEC
RANDOM_STATE = 42

# ==============================
# FOCUS FEATURE EXTRACTOR
# ==============================
def extract_focus_features(eeg_window, fs=250):
    freqs, psd = welch(eeg_window, fs=fs, nperseg=256)

    alpha = (freqs >= 8) & (freqs <= 12)
    beta  = (freqs >= 13) & (freqs <= 30)
    theta = (freqs >= 4) & (freqs <= 7)

    alpha_p = np.trapezoid(psd[alpha], freqs[alpha])
    beta_p  = np.trapezoid(psd[beta], freqs[beta])
    theta_p = np.trapezoid(psd[theta], freqs[theta])

    alpha_beta_ratio = alpha_p / (beta_p + 1e-10)
    signal_variance = np.var(eeg_window)
    attention_index = beta_p / (alpha_p + theta_p + 1e-10)

    return [
        alpha_p,
        beta_p,
        theta_p,
        alpha_beta_ratio,
        signal_variance,
        attention_index
    ]

# ==============================
# LOAD BONN DATA
# ==============================
def load_bonn_dataset(base_path):
    X, y = [], []

    class_map = {
        "A": 0, "B": 0,  # Non-seizure
        "C": 0, "D": 0,
        "E": 1           # Seizure
    }

    for folder, label in class_map.items():
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            signal = np.loadtxt(os.path.join(folder_path, file))

            # Resample to 250 Hz
            new_len = int(len(signal) * TARGET_FS / ORIG_FS)
            signal_250 = resample(signal, new_len)

            # Segment into 2-sec windows
            for i in range(0, len(signal_250) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                window = signal_250[i:i + WINDOW_SAMPLES]
                features = extract_focus_features(window)
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("\n🔍 Loading Bonn dataset...")
    X, y = load_bonn_dataset(BONN_PATH)

    print(f"Total windows: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Train/test split (NO leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    print("\n🚀 Training XGBoost...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📊 Evaluation Results")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Normal", "Seizure"]))

    # Save
    joblib.dump(model, "seizure_xgb_focus.pkl")
    joblib.dump(scaler, "seizure_focus_scaler.pkl")

    print("\n✅ Model saved:")
    print("   seizure_xgb_focus.pkl")
    print("   seizure_focus_scaler.pkl")
