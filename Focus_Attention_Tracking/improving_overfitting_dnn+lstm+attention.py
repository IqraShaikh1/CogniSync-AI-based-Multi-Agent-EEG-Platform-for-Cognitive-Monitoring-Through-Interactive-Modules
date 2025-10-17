import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from xgboost import XGBClassifier
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ===============================
# 1. Load Data
# ===============================
df = pd.read_csv("extended_features.csv")
exclude_cols = ["participant", "activity", "label"]
X = df.drop(columns=exclude_cols)
y = df["label"]

# Fill missing values
X = X.fillna(X.median())

# Remove low-variance features
selector = VarianceThreshold(threshold=1e-3)
X_var = selector.fit_transform(X)
X_var_df = pd.DataFrame(X_var, columns=X.columns[selector.get_support()])

# Remove highly correlated features
corr_matrix = X_var_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_sel = X_var_df.drop(columns=to_drop)

# Clip outliers
for col in X_sel.columns:
    lower, upper = X_sel[col].quantile([0.05, 0.95])
    X_sel[col] = X_sel[col].clip(lower, upper)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# Feature selection
fs = SelectKBest(f_classif, k=20)
X_fs = fs.fit_transform(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size=0.2, stratify=y, random_state=42)

# ===============================
# 2. Train Models
# ===============================
model_scores = {}
best_model = None
best_model_name = ""
best_acc = 0.0

# ---- RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
acc = accuracy_score(y_test, rf_preds)
model_scores["RandomForest"] = acc
print(f"RandomForest Accuracy: {acc:.4f}")
print(classification_report(y_test, rf_preds))

# ---- XGBoost
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
acc = accuracy_score(y_test, xgb_preds)
model_scores["XGBoost"] = acc
print(f"XGBoost Accuracy: {acc:.4f}")
print(classification_report(y_test, xgb_preds))

# ---- SVM
svc = SVC(probability=True, random_state=42)
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_test)
acc = accuracy_score(y_test, svc_preds)
model_scores["SVM"] = acc
print(f"SVM Accuracy: {acc:.4f}")
print(classification_report(y_test, svc_preds))

# ---- Stacking
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('xgb', XGBClassifier(eval_metric='mlogloss', random_state=42))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
stack.fit(X_train, y_train)
stack_preds = stack.predict(X_test)
acc = accuracy_score(y_test, stack_preds)
model_scores["Stacking"] = acc
print(f"Stacking Accuracy: {acc:.4f}")
print(classification_report(y_test, stack_preds))

# ===============================
# 3. Transformer (PyTorch)
# ===============================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EEGTransformer(nn.Module):
    def __init__(self, n_channels, seq_len, n_classes, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_classes)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

X_train_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

train_dataset = EEGDataset(X_train_t, y_train)
test_dataset = EEGDataset(X_test_t, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EEGTransformer(n_channels=X_train.shape[1], seq_len=1, n_classes=len(np.unique(y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(25):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Transformer evaluation
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        out = model(xb)
        preds.append(out.argmax(dim=1).cpu())
preds = torch.cat(preds).numpy()
acc = accuracy_score(y_test, preds)
model_scores["Transformer"] = acc
print(f"EEG Transformer Accuracy: {acc:.4f}")
print(classification_report(y_test, preds))

# ===============================
# 4. Save Best Model
# ===============================
best_model_name = max(model_scores, key=model_scores.get)
best_acc = model_scores[best_model_name]

print("\n==============================")
print(f"Best Model: {best_model_name} with Accuracy = {best_acc:.4f}")
print("==============================")

# Save model based on type
if best_model_name == "RandomForest":
    joblib.dump(rf, "best_model_randomforest_H.pkl")
elif best_model_name == "XGBoost":
    joblib.dump(xgb, "best_model_xgboost_h.pkl")
elif best_model_name == "SVM":
    joblib.dump(svc, "best_model_svm_H.pkl")
elif best_model_name == "Stacking":
    joblib.dump(stack, "best_model_stacking_H.pkl")
elif best_model_name == "Transformer":
    torch.save(model.state_dict(), "best_model_transformer_h.pth")

print(f"✅ Saved best model: {best_model_name}")
