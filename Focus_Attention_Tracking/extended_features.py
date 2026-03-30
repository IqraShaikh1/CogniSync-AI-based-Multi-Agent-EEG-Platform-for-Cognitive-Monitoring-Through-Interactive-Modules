import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from antropy import sample_entropy, perm_entropy, hjorth_params

# ---------------- SETTINGS ----------------
CSV_PATH = r"C:\Users\major\Desktop\eeg data train\Focus_Attention_Tracking\merged_eeg_features.csv"
OUTPUT_PATH = "extended_features.csv"
WINDOW_SIZE = 500   # 2 sec window (250 Hz x 2 sec)
STEP_SIZE = 250     # 50% overlap
# ------------------------------------------

print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Keep only valid activities
valid_activities = ["baseline", "focus1", "focus2", "distraction"]
df = df[df["activity"].isin(valid_activities)]

# Encode labels
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["activity"])
print("✅ Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# Features to aggregate
basic_features = ["alpha_power", "beta_power", "theta_power",
                  "alpha_beta_ratio", "signal_variance", "attention_index"]

# ---------------- HELPER FUNCTIONS ----------------
def compute_window_features(window_df):
    feats = {}

    # Raw EEG (for nonlinear features)
    eeg_vals = window_df["eeg_value"].values

    # ---- Band Powers ----
    alpha = window_df["alpha_power"].values
    beta = window_df["beta_power"].values
    theta = window_df["theta_power"].values

    # Ratios
    feats["theta_beta_ratio"] = np.mean(theta) / (np.mean(beta) + 1e-6)
    feats["alpha_theta_ratio"] = np.mean(alpha) / (np.mean(theta) + 1e-6)
    feats["alpha_plus_theta_beta"] = (np.mean(alpha) + np.mean(theta)) / (np.mean(beta) + 1e-6)

    # ---- Stats on each feature ----
    for f in basic_features:
        vals = window_df[f].values
        feats[f+"_mean"] = np.mean(vals)
        feats[f+"_std"] = np.std(vals)
        feats[f+"_min"] = np.min(vals)
        feats[f+"_max"] = np.max(vals)
        feats[f+"_skew"] = skew(vals)
        feats[f+"_kurt"] = kurtosis(vals)

    # ---- Non-linear features ----
    try:
        feats["sample_entropy"] = sample_entropy(eeg_vals)
    except:
        feats["sample_entropy"] = 0

    try:
        feats["perm_entropy"] = perm_entropy(eeg_vals, normalize=True)
    except:
        feats["perm_entropy"] = 0

    try:
        hj_activity, hj_mobility, hj_complexity = hjorth_params(eeg_vals)
        feats["hjorth_activity"] = hj_activity
        feats["hjorth_mobility"] = hj_mobility
        feats["hjorth_complexity"] = hj_complexity
    except:
        feats["hjorth_activity"] = feats["hjorth_mobility"] = feats["hjorth_complexity"] = 0

    return feats
# --------------------------------------------------

print("⚙️ Computing windowed features...")
X, y = [], []
step_count = 0

for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
    window = df.iloc[i:i+WINDOW_SIZE]

    feats = compute_window_features(window)
    feats["label"] = window["label"].iloc[-1]
    feats["participant"] = window["participant"].iloc[-1]
    feats["activity"] = window["activity"].iloc[-1]

    X.append(feats)
    y.append(window["label"].iloc[-1])

    step_count += 1
    if step_count % 500 == 0:
        print(f"Processed {step_count} windows...")

# Convert to DataFrame
feature_df = pd.DataFrame(X)
print("✅ Final feature shape:", feature_df.shape)

# Save
feature_df.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Saved engineered features to {OUTPUT_PATH}")
