import os
import glob
import pandas as pd

# -------- SETTINGS --------
DATA_DIR = r"C:\Users\major\Desktop\eeg data train\Focus_Attention_Tracking"
OUTPUT_FILE = r"C:\Users\major\Desktop\eeg data train\merged_eeg_features.csv"
# --------------------------

all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

merged_data = []

for file in all_files:
    filename = os.path.basename(file)
    # Example: AdityaSEAComps_baseline_20250813_152054.csv
    parts = filename.split("_")

    if len(parts) < 3:
        print(f"⚠️ Skipping {filename}, unexpected format")
        continue

    participant = parts[0]              # e.g., "AdityaSEAComps"
    activity = parts[1].lower()         # e.g., "baseline"
    date_time = "_".join(parts[2:])     # e.g., "20250813_152054.csv"

    try:
        df = pd.read_csv(file)
        df["participant"] = participant
        df["activity"] = activity
        df["file"] = filename
        merged_data.append(df)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

if merged_data:
    final_df = pd.concat(merged_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Merged {len(all_files)} files into {OUTPUT_FILE}")
    print(f"Final dataset shape: {final_df.shape}")
else:
    print("⚠️ No valid data merged.")
