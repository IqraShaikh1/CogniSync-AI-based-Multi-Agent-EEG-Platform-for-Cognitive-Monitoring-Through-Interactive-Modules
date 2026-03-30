import pandas as pd
import glob
import os

# ===== MERGE ALL CSV FILES =====

# Path to your CSV files directory
DATA_DIR = "."  # Change this to your data folder path if different
OUTPUT_FILE = "merged_eeg_emotions.csv"

# Find all CSV files matching the pattern
csv_files = glob.glob(os.path.join(DATA_DIR, "P*_S*_*.csv"))

print(f"Found {len(csv_files)} CSV files to merge")
print("\nFiles found:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# Read and concatenate all CSV files
dfs = []
for i, csv_file in enumerate(csv_files, 1):
    print(f"\nReading file {i}/{len(csv_files)}: {os.path.basename(csv_file)}")
    try:
        df = pd.read_csv(csv_file)
        print(f"  ✓ Loaded {len(df)} rows")
        dfs.append(df)
    except Exception as e:
        print(f"  ✗ Error reading {csv_file}: {e}")

# Concatenate all dataframes
print(f"\n{'='*60}")
print("Merging all dataframes...")
merged_df = pd.concat(dfs, ignore_index=True)

print(f"\n✅ Merge completed!")
print(f"📊 Total rows: {len(merged_df):,}")
print(f"📊 Total columns: {len(merged_df.columns)}")
print(f"\n📈 Emotion distribution:")
print(merged_df['emotion_label'].value_counts().sort_index())
print(f"\n👥 Participant distribution:")
print(merged_df['participant_id'].value_counts().sort_index())

# Save merged dataset
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Merged dataset saved to: {OUTPUT_FILE}")
print(f"💾 File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")

# Display first few rows
print(f"\n{'='*60}")
print("Preview of merged dataset:")
print(merged_df.head())

# Check for missing values
print(f"\n{'='*60}")
print("Missing values check:")
missing = merged_df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print("⚠️  Missing values detected:")
    print(missing[missing > 0])