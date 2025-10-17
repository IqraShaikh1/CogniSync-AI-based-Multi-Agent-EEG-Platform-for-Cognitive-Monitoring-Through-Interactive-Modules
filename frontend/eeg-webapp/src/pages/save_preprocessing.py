import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import joblib

# ===============================
# Recreate preprocessing pipeline and save objects
# ===============================
print("Loading data...")
df = pd.read_csv("extended_features.csv")
exclude_cols = ["participant", "activity", "label"]
X = df.drop(columns=exclude_cols)
y = df["label"]

# Fill missing values
print("Filling missing values...")
X = X.fillna(X.median())

# Remove low-variance features
print("Removing low-variance features...")
selector = VarianceThreshold(threshold=1e-3)
X_var = selector.fit_transform(X)
X_var_df = pd.DataFrame(X_var, columns=X.columns[selector.get_support()])

# Remove highly correlated features
print("Removing correlated features...")
corr_matrix = X_var_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_sel = X_var_df.drop(columns=to_drop)

# Save selected feature names after correlation removal
selected_features = X_sel.columns.tolist()

# Clip outliers
print("Clipping outliers...")
for col in X_sel.columns:
    lower, upper_val = X_sel[col].quantile([0.05, 0.95])
    X_sel[col] = X_sel[col].clip(lower, upper_val)

# Scaling
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# Feature selection
print("Selecting top features...")
fs = SelectKBest(f_classif, k=20)
X_fs = fs.fit_transform(X_scaled, y)

# Save all preprocessing objects
print("\nSaving preprocessing objects...")
joblib.dump(selector, "variance_selector.pkl")
print("✅ Saved: variance_selector.pkl")

joblib.dump(selected_features, "selected_features.pkl")
print("✅ Saved: selected_features.pkl")

joblib.dump(scaler, "scaler.pkl")
print("✅ Saved: scaler.pkl")

joblib.dump(fs, "feature_selector.pkl")
print("✅ Saved: feature_selector.pkl")

print("\n" + "="*50)
print("All preprocessing objects saved successfully!")
print("="*50)
print("\nSummary:")
print(f"  - Original features: {len(X.columns)}")
print(f"  - After variance filter: {len(X_var_df.columns)}")
print(f"  - After correlation filter: {len(selected_features)}")
print(f"  - Final selected features: 20")
print("\nYou can now run app.py to start the API server.")