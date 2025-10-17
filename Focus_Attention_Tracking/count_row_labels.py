import pandas as pd

# Load merged dataset
df = pd.read_csv(r"C:\Users\major\Desktop\eeg data train\merged_eeg_features_clean.csv")

# Count rows for each activity
label_counts = df["activity"].value_counts()

print("Row counts per activity:")
print(label_counts)

# Optional: also show % distribution
print("\nPercentage distribution:")
print((label_counts / len(df)) * 100)
