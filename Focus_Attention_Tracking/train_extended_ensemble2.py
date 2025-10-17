import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# ---------------- SETTINGS ----------------
CSV_PATH = "extended_features2.csv"
LOG_PATH = "training_log2.csv"
# ------------------------------------------

print("📂 Loading engineered dataset...")
df = pd.read_csv(CSV_PATH)

# Extract features and labels
exclude_cols = ["participant", "activity", "label"]
X = df.drop(columns=exclude_cols).values
y = df["label"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define base models
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss"
)
lgbm_clf = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=-1,
    num_leaves=64, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)

# Stacking ensemble
estimators = [
    ("rf", rf),
    ("xgb", xgb_clf),
    ("lgbm", lgbm_clf)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=200, random_state=42),
    n_jobs=-1
)

print("🚀 Training ensemble model...")
stacking_clf.fit(X_train, y_train)

# Evaluate
y_pred = stacking_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}\n")

print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(df["activity"]),
            yticklabels=np.unique(df["activity"]))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODELS ----------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Add "new_" prefix to indicate these are new retrained models
MODEL_PATH = f"new_eeg_extended_ensemble_{timestamp}.pkl"
SCALER_PATH = f"new_eeg_scaler_{timestamp}.pkl"
ENCODER_PATH = f"new_eeg_labelencoder_{timestamp}.pkl"

# Save model, scaler, encoder
joblib.dump(stacking_clf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

encoder = LabelEncoder()
encoder.fit(df["activity"])
joblib.dump(encoder, ENCODER_PATH)

print(f"\n💾 Saved retrained model as:")
print(f"   {MODEL_PATH}")
print(f"   {SCALER_PATH}")
print(f"   {ENCODER_PATH}")


# ---------------- LOGGING ----------------
log_entry = {
    "timestamp": timestamp,
    "accuracy": acc,
    "model_file": MODEL_PATH,
    "scaler_file": SCALER_PATH,
    "encoder_file": ENCODER_PATH
}

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
else:
    log_df = pd.DataFrame([log_entry])

log_df.to_csv(LOG_PATH, index=False)
print(f"📝 Training log updated: {LOG_PATH}")
