import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------- SETTINGS --------
CSV_PATH = r"C:\Users\major\Desktop\eeg data train\Focus_Attention_Tracking\merged_eeg_features.csv"
WINDOW_SIZE = 500   # 2 sec window (250 Hz x 2 sec)
STEP_SIZE = 250     # 50% overlap (1 sec step)
# --------------------------

# 1. Load dataset
df = pd.read_csv(CSV_PATH)

# 2. Keep only valid activities
valid_activities = ["baseline", "focus1", "focus2", "distraction"]
df = df[df["activity"].isin(valid_activities)]

# 3. Encode activity labels
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["activity"])

print("✅ Activities used:", df["activity"].unique())
print("✅ Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# 4. Select features
features = ["alpha_power", "beta_power", "theta_power", 
            "alpha_beta_ratio", "signal_variance", "attention_index"]
X = df[features].values
y = df["label"].values

# 5. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 6. Convert into sliding windows
X_seq, y_seq = [], []
for i in range(0, len(X) - WINDOW_SIZE, STEP_SIZE):
    X_seq.append(X[i:i+WINDOW_SIZE])
    y_seq.append(y[i+WINDOW_SIZE-1])  # label at end of window

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Shape of sequences:", X_seq.shape, "Labels:", y_seq.shape)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# 8. Save preprocessed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
