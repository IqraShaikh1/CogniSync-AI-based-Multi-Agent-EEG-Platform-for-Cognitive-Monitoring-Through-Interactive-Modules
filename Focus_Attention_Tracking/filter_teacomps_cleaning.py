# clean_dataset.py
import pandas as pd

# 1. Load your merged dataset
file_path = r"C:\Users\major\Desktop\eeg data train\Focus_Attention_Tracking\merged_eeg_features.csv"

df = pd.read_csv(file_path)

print("✅ Original shape:", df.shape)

# 2. Define the valid activity labels
valid_labels = ["baseline", "distraction", "focus1", "focus2"]

# 3. Inspect current unique labels
print("\n🔎 Unique activity labels found:")
print(df["activity"].unique())

# 4. Filter dataset to keep only valid labels
df_clean = df[df["activity"].isin(valid_labels)].copy()

print("\n✅ Cleaned shape:", df_clean.shape)

# 5. Show counts per activity
label_counts = df_clean["activity"].value_counts()
print("\n📊 Row counts per activity:")
print(label_counts)

# 6. Show percentage distribution
label_percent = df_clean["activity"].value_counts(normalize=True) * 100
print("\n📊 Percentage distribution:")
print(label_percent)

# 7. Save to a new CSV (don’t overwrite the old one)
output_file = "merged_eeg_features_clean.csv"
df_clean.to_csv(output_file, index=False)

print(f"\n💾 Cleaned dataset saved as: {output_file}")
