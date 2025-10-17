# eeg_ensemble_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

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

X_seq = np.array(X_seq)   # shape: (num_windows, 500, 6)
y_seq = np.array(y_seq)

print("Shape of sequences:", X_seq.shape, "Labels:", y_seq.shape)

# 7. Aggregate statistical features per window
X_feat = []
for window in X_seq:
    stats = []
    for col in range(window.shape[1]):
        col_data = window[:, col]
        stats.extend([
            np.mean(col_data),
            np.std(col_data),
            np.min(col_data),
            np.max(col_data),
            np.median(col_data),
            pd.Series(col_data).skew(),
            pd.Series(col_data).kurt()
        ])
    X_feat.append(stats)

X_feat = np.array(X_feat)

print("Aggregated feature shape:", X_feat.shape)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# 9. Define models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    tree_method="hist")  # for speed
svm = SVC(probability=True, kernel="rbf", C=10, gamma="scale", random_state=42)

# 10. Ensemble with stacking
estimators = [
    ('rf', rf),
    ('xgb', xgb),
    ('svm', svm)
]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=5,
    n_jobs=-1
)

# 11. Train
print("🚀 Training ensemble model...")
stack_model.fit(X_train, y_train)

# 12. Evaluate
y_pred = stack_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Accuracy:", acc)
print("\n📊 Confusion Matrix:\n", cm)
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

# 13. Save model, scaler, and encoder
joblib.dump(stack_model, "eeg_ensemble_model.pkl")
joblib.dump(scaler, "eeg_scaler.pkl")
joblib.dump(encoder, "eeg_labelencoder.pkl")

print("\n💾 Model, Scaler, and Encoder saved successfully!")
