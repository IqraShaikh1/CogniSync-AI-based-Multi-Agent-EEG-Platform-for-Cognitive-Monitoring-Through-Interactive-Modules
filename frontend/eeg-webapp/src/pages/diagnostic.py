# diagnostic.py - Check what features the scaler expects
import joblib
import numpy as np

print("="*60)
print("CHECKING SCALER CONFIGURATION")
print("="*60)

# Load the scaler
scaler = joblib.load("mental_health_scaler.joblib")

print(f"\n✅ Scaler loaded successfully")
print(f"Expected number of features: {scaler.n_features_in_}")

# Check if scaler has feature names
if hasattr(scaler, 'feature_names_in_'):
    print(f"\n📋 Feature names used during training:")
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"  {i+1}. {name}")
else:
    print(f"\n⚠️ Scaler doesn't have feature names stored")
    print(f"The scaler expects {scaler.n_features_in_} features")

# Print scaler statistics
print(f"\n📊 Scaler Statistics:")
print(f"Mean values: {scaler.mean_}")
print(f"Scale values: {scaler.scale_}")

print("\n" + "="*60)
print("Based on your dataset columns, the 7 features should be:")
print("="*60)
print("1. timestamp_ms")
print("2. eeg_value") 
print("3. alpha_power")
print("4. beta_power")
print("5. theta_power")
print("6. alpha_beta_ratio")
print("7. signal_variance")
print("8. attention_index")
print("\nSince we have 7 features expected, likely these are used:")
print("1. alpha_power")
print("2. beta_power")
print("3. theta_power")
print("4. alpha_beta_ratio")
print("5. signal_variance")
print("6. attention_index")
print("7. ??? (one more feature - possibly delta_power, theta_beta_ratio, or another derived feature)")
print("="*60)