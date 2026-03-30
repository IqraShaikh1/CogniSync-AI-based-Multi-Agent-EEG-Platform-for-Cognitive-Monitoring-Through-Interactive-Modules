import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
MERGED_FILE = "merged_eeg_emotions.csv"
MODEL_FILENAME = "eeg_emotion_xgb_model.pkl"
SCALER_FILENAME = "eeg_scaler_emotion2.pkl"
LABEL_ENCODER_FILENAME = "eeg_label_encoder-EMOTION2.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("="*60)
print("EEG EMOTION RECOGNITION - FAST XGBoost")
print("="*60)

# ===== LOAD DATA =====
print(f"\n📂 Loading data...")
df = pd.read_csv(MERGED_FILE)
print(f"✓ {len(df):,} samples")

# ===== PREPARE DATA =====
metadata_cols = ['timestamp_ms', 'eeg_raw_value', 'emotion_label', 
                 'participant_id', 'session_number']
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols].values
y = df['emotion_label'].values
X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
emotion_classes = label_encoder.classes_

print(f"✓ {len(feature_cols)} features | {len(emotion_classes)} emotions")

# Check distribution
print(f"\n📊 Distribution:")
for emotion in emotion_classes:
    count = np.sum(y == emotion)
    print(f"  {emotion:12s}: {count:,}")

# ===== SPLIT & SCALE =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\n🔀 Train: {len(X_train):,} | Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = class_weights[y_train]

# ===== TRAIN XGBOOST (FAST & ACCURATE) =====
print(f"\n🚀 Training XGBoost...")

model = XGBClassifier(
    n_estimators=100,        # Fast
    max_depth=6,             # Shallow = faster
    learning_rate=0.2,       # Higher = faster convergence
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',      # Fastest method
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train_scaled, y_train, sample_weight=sample_weights, verbose=False)
print("✓ Complete!")

# ===== EVALUATE =====
print(f"\n📈 RESULTS")
print("="*60)

y_test_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%\n")

# Classification report
print(classification_report(y_test, y_test_pred, target_names=emotion_classes))

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_test_pred)

print(f"📉 Confusion Matrix:")
print(cm)

print(f"\n✅ Per-Emotion Accuracy:")
for i, emotion in enumerate(emotion_classes):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = (correct / total * 100) if total > 0 else 0
    status = "✓" if acc >= 80 else "⚠️"
    print(f"  {status} {emotion:12s}: {acc:.2f}% ({correct}/{total})")

# ===== PLOT =====
plt.figure(figsize=(10, 8))

# Subplot 1: Raw counts
plt.subplot(2, 1, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.title(f'Confusion Matrix (Accuracy: {test_acc*100:.2f}%)', fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')

# Subplot 2: Normalized
plt.subplot(2, 1, 2)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.title('Per-Emotion Accuracy (%)', fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print(f"\n💾 Saved: confusion_matrix.png")

# ===== SAVE MODEL =====
print(f"\n💾 Saving model...")
joblib.dump(model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)
joblib.dump(label_encoder, LABEL_ENCODER_FILENAME)
print(f"✓ {MODEL_FILENAME}")
print(f"✓ {SCALER_FILENAME}")
print(f"✓ {LABEL_ENCODER_FILENAME}")

print(f"\n{'='*60}")
print(f"✅ DONE! Accuracy: {test_acc*100:.2f}%")
print(f"{'='*60}")