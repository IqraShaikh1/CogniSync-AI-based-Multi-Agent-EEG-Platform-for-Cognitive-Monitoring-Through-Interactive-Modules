# preprocess_long_window.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# SETTINGS
CSV_PATH = r"C:\Users\major\Desktop\eeg data train\Focus_Attention_Tracking\merged_eeg_features.csv"
OUT_DIR = r"C:\Users\major\Desktop\eeg data train"   # change if desired
WINDOW_SEC = 4.0           # 4 seconds
SAMPLE_RATE = 250          # samples per second
WINDOW_SIZE = int(WINDOW_SEC * SAMPLE_RATE)  # 1000
STEP_SIZE = int(WINDOW_SIZE // 2)           # 50% overlap = 500

# Valid activities (filter)
VALID_ACTIVITIES = ["baseline", "focus1", "distraction", "focus2"]

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

# filter unexpected activities
df = df[df["activity"].isin(VALID_ACTIVITIES)].copy()
if df.empty:
    raise SystemExit("No valid rows found after filtering activities.")

print("Activities present:", df["activity"].unique())

# Encode labels and save encoder
le = LabelEncoder()
df["label"] = le.fit_transform(df["activity"])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))

# Features to use
features = ["alpha_power", "beta_power", "theta_power",
            "alpha_beta_ratio", "signal_variance", "attention_index"]

# Drop rows with NaNs in features
df = df.dropna(subset=features)

# Fit scaler on features (important to save scaler)
scaler = StandardScaler()
X_all = scaler.fit_transform(df[features].values)
joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
print("Scaler saved.")

y_all = df["label"].values
timestamps = df["timestamp_ms"].values  # optional, for debugging

# Convert to sliding windows (this keeps everything in memory)
print("Converting to sliding windows (window_size=", WINDOW_SIZE, "step=", STEP_SIZE, ")")
X_seq = []
y_seq = []

n_rows = X_all.shape[0]
# iterate windows
i = 0
while i + WINDOW_SIZE <= n_rows:
    window_X = X_all[i:i + WINDOW_SIZE]
    # label for window = label at window end (same as before)
    window_y = y_all[i + WINDOW_SIZE - 1]
    X_seq.append(window_X)
    y_seq.append(window_y)
    i += STEP_SIZE

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.int32)

print("Created windows:", X_seq.shape, "labels:", y_seq.shape)

# Train/test split (stratify by label)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.20, random_state=42, stratify=y_seq
)

# Save arrays
np.save(os.path.join(OUT_DIR, "X_train_long.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_test_long.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_train_long.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test_long.npy"), y_test)

print("Saved X_train_long.npy, X_test_long.npy, y_train_long.npy, y_test_long.npy")
print("DONE.")
