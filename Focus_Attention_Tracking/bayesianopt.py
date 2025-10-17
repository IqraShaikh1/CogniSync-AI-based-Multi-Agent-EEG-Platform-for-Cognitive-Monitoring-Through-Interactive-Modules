import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import optuna

# ---------------- SETTINGS ----------------
CSV_PATH = "extended_features.csv"
LOG_PATH = "training_log_dnn_bilstm_optuna.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TRIALS = 25  # Adjust for hyperparameter search depth
BATCH_SIZE = 128
# ------------------------------------------

print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)

# ---------------- Features and Labels ----------------
exclude_cols = ["participant", "activity", "label"]
X = df.drop(columns=exclude_cols, errors='ignore').values
y = df["label"].values

# Handle NaNs
X = np.nan_to_num(X, nan=0.0)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Convert encoder classes to strings for reporting
classes_str = [str(c) for c in encoder.classes_]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Windowed Sequences ----------------
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

# ---------------- PyTorch Dataset ----------------
class EEGSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- Model Definition ----------------
class DNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, num_classes, dnn_hidden=128, lstm_hidden=128, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, dnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dnn_hidden, dnn_hidden),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=dnn_hidden, hidden_size=lstm_hidden, 
                            num_layers=lstm_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(lstm_hidden*2, 1)
        self.fc = nn.Linear(lstm_hidden*2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, feat = x.size()
        x = x.view(batch_size*seq_len, feat)
        x = self.dnn(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        # Attention
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq_len, 1]
        x = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        x = self.dropout(x)
        out = self.fc(x)
        return out

# ---------------- Objective Function for Optuna ----------------
def objective(trial):
    seq_len = trial.suggest_int("seq_len", 3, 10)
    dnn_hidden = trial.suggest_categorical("dnn_hidden", [64, 128, 256])
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 128, 256])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    X_seq, y_seq = create_sequences(X_scaled, y, seq_len)
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
    )
    
    train_dataset = EEGSequenceDataset(X_train, y_train)
    val_dataset = EEGSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DNN_BiLSTM_Attention(input_dim=X.shape[1], num_classes=len(np.unique(y)),
                                 dnn_hidden=dnn_hidden, lstm_hidden=lstm_hidden, 
                                 lstm_layers=lstm_layers, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    EPOCHS = 15  # Keep small for tuning
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    
    # Validation accuracy
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred_list.extend(preds)
            y_true_list.extend(yb.numpy())
    acc = accuracy_score(y_true_list, y_pred_list)
    return acc

# ---------------- Run Optuna Study ----------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=MAX_TRIALS)
print("✅ Best trial:")
print(study.best_trial.params)

# ---------------- Train Final Model with Best Params ----------------
best_params = study.best_trial.params
SEQ_LEN = best_params["seq_len"]
X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
)
train_dataset = EEGSequenceDataset(X_train, y_train)
test_dataset = EEGSequenceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = DNN_BiLSTM_Attention(input_dim=X.shape[1], num_classes=len(np.unique(y)),
                             dnn_hidden=best_params["dnn_hidden"],
                             lstm_hidden=best_params["lstm_hidden"],
                             lstm_layers=best_params["lstm_layers"],
                             dropout=best_params["dropout"]).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

EPOCHS = 25
print("🚀 Training final model with best hyperparameters...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader.dataset):.4f}")

# ---------------- Evaluation ----------------
model.eval()
y_pred_list, y_true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred_list.extend(preds)
        y_true_list.extend(yb.numpy())

acc = accuracy_score(y_true_list, y_pred_list)
print(f"\n✅ Final Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true_list, y_pred_list)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes_str,
            yticklabels=classes_str)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true_list, y_pred_list, target_names=classes_str))

# ---------------- Save Model ----------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = f"dnn_bilstm_attn_eeg_{timestamp}.pt"
SCALER_PATH = f"dnn_bilstm_scaler_{timestamp}.pkl"
ENCODER_PATH = f"dnn_bilstm_labelencoder_{timestamp}.pkl"

torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(encoder, ENCODER_PATH)

print(f"\n💾 Saved final model:")
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
