import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
CSV_PATH = "extended_features.csv"
LOG_PATH = "training_log_tuned.csv"
SAMPLE_SIZE = 200000  # optional subset for faster tuning
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

# Optional subset for tuning
if SAMPLE_SIZE and SAMPLE_SIZE < len(X):
    idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
    X_tune, y_tune = X[idx], y[idx]
else:
    X_tune, y_tune = X, y

X_train, X_test, y_train, y_test = train_test_split(
    X_tune, y_tune, test_size=0.2, random_state=42, stratify=y_tune
)

print("✅ Data ready for tuning, shape:", X_train.shape)

# ---------------- GridSearchCV for RandomForest ----------------
print("🔧 Tuning RandomForest...")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="accuracy", verbose=2)
rf_grid.fit(X_train, y_train)
print("✅ Best RF params:", rf_grid.best_params_)

# ---------------- GridSearchCV for XGBoost ----------------
print("🔧 Tuning XGBoost...")
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="mlogloss", random_state=42, n_jobs=-1
)
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_grid = GridSearchCV(xgb_clf, xgb_params, cv=3, scoring="accuracy", verbose=2)
xgb_grid.fit(X_train, y_train)
print("✅ Best XGB params:", xgb_grid.best_params_)

# ---------------- GridSearchCV for LightGBM ----------------
print("🔧 Tuning LightGBM...")
lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
lgb_params = {
    "n_estimators": [100, 200],
    "num_leaves": [31, 64],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

lgb_grid = GridSearchCV(lgb_clf, lgb_params, cv=3, scoring="accuracy", verbose=2)
lgb_grid.fit(X_train, y_train)
print("✅ Best LGB params:", lgb_grid.best_params_)

# ---------------- Retrain stacking on full dataset ----------------
print("🚀 Training tuned stacking ensemble on full dataset...")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_tuned = RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=-1)
xgb_tuned = xgb.XGBClassifier(**xgb_grid.best_params_, use_label_encoder=False,
                              eval_metric="mlogloss", n_jobs=-1, random_state=42)
lgb_tuned = lgb.LGBMClassifier(**lgb_grid.best_params_, random_state=42, n_jobs=-1)

estimators = [
    ("rf_tuned", rf_tuned),
    ("xgb_tuned", xgb_tuned),
    ("lgb_tuned", lgb_tuned)
]

stacking_tuned = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=200, random_state=42),
    n_jobs=-1
)

stacking_tuned.fit(X_train_full, y_train_full)

# ---------------- Evaluate ----------------
y_pred_full = stacking_tuned.predict(X_test_full)
acc = accuracy_score(y_test_full, y_pred_full)
print(f"\n✅ Accuracy on full dataset: {acc:.4f}\n")

cm = confusion_matrix(y_test_full, y_pred_full)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(df["activity"]),
            yticklabels=np.unique(df["activity"]))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\n📋 Classification Report:")
print(classification_report(y_test_full, y_pred_full))

# ---------------- Save tuned models ----------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = f"tuned_eeg_stacking_{timestamp}.pkl"
SCALER_PATH = f"tuned_eeg_scaler_{timestamp}.pkl"
ENCODER_PATH = f"tuned_eeg_labelencoder_{timestamp}.pkl"

joblib.dump(stacking_tuned, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
encoder = LabelEncoder()
encoder.fit(df["activity"])
joblib.dump(encoder, ENCODER_PATH)

print(f"\n💾 Saved tuned models:")
print(f"   {MODEL_PATH}")
print(f"   {SCALER_PATH}")
print(f"   {ENCODER_PATH}")

# ---------------- Logging ----------------
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
