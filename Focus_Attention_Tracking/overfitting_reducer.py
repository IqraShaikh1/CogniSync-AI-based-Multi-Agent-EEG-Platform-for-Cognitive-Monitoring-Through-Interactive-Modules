import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Load Dataset ----------------
CSV_PATH = "extended_features.csv"
df = pd.read_csv(CSV_PATH)

# ---------------- Participant-wise Split ----------------
participants = df["participant"].unique()
np.random.seed(42)
test_participants = np.random.choice(participants, size=int(0.2*len(participants)), replace=False)

train_idx = df["participant"].isin([p for p in participants if p not in test_participants])
test_idx = df["participant"].isin(test_participants)

train_df = df[train_idx]
test_df = df[test_idx]

# ---------------- Features & Labels ----------------
exclude_cols = ["participant", "activity", "label"]
X_train = train_df.drop(columns=exclude_cols, errors='ignore').values
y_train = train_df["label"].values

X_test = test_df.drop(columns=exclude_cols, errors='ignore').values
y_test = test_df["label"].values

# Handle NaNs
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Encode labels
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Train Random Forest ----------------
clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
clf.fit(X_train_scaled, y_train_enc)

# ---------------- Evaluation ----------------
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test_enc, y_pred)
print(f"✅ Participant-wise Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))
