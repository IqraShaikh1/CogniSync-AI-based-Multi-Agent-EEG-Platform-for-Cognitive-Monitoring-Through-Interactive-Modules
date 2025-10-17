#!/usr/bin/env python3
"""
EEG Mental Health Detection — Simplified Data Collection

This script:
- Prompts for participant details (name, department).
- Guides the participant through a fixed sequence of tasks:
  Baseline_Open, Cognitive_Stress, Social_Stress, Reading, Relaxation.
- Records EEG from Serial (ESP32/Arduino) and computes features:
  alpha_power, beta_power, theta_power, alpha_beta_ratio,
  signal_variance, attention_index.
- Saves one CSV per task for the same individual inside a folder named after the participant.

File format:
  data/<participant_name>_<department>/<task_label>_<timestamp>.csv

Dependencies:
  pip install numpy scipy pandas pyserial
"""
import os
import time
import csv
import serial
import numpy as np
from datetime import datetime
from scipy.signal import welch

# -------- SETTINGS --------
PORT = "COM6"       # Change to match your system
BAUD = 115200
SAMPLE_RATE = 250   # Hz (should match Arduino/device)
WINDOW_SEC = 2      # Window length for feature computation
# --------------------------

# -------- TASKS --------
TASKS = [
    {"label": "Baseline_Open", "duration": 120, "instructions": "Sit relaxed, eyes OPEN, breathe normally."},
    {"label": "Cognitive_Stress", "duration": 120, "instructions": "Do quick mental arithmetic out loud or silently."},
    {"label": "Social_Stress", "duration": 120, "instructions": "Prepare 30s, then speak 90s as if presenting."},
    {"label": "Reading", "duration": 120, "instructions": "Silently read a neutral paragraph, stay engaged."},
    {"label": "Relaxation", "duration": 120, "instructions": "Follow paced breathing: inhale 4s, exhale 6s."},
]
# ------------------------
# Ask for participant details
participant = input("Enter participant name: ").strip()
department = input("Enter department: ").strip()

# Create folder for this participant
base_dir = f"data/{participant}_{department}"
os.makedirs(base_dir, exist_ok=True)

# Connect to EEG device
ser = serial.Serial(PORT, BAUD)
print(f"Connected to {PORT} at {BAUD} baud")

# Function to compute features
def compute_features(eeg_window):
    freqs, psd = welch(eeg_window, fs=SAMPLE_RATE, nperseg=256)
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    beta_mask  = (freqs >= 13) & (freqs <= 30)
    theta_mask = (freqs >= 4) & (freqs <= 7)

    alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
    beta_power  = np.trapezoid(psd[beta_mask], freqs[beta_mask])
    theta_power = np.trapezoid(psd[theta_mask], freqs[theta_mask])

    alpha_beta_ratio = alpha_power / beta_power if beta_power != 0 else 0
    signal_variance = np.var(eeg_window)
    attention_index = beta_power / (alpha_power + theta_power + 1e-6)

    return alpha_power, beta_power, theta_power, alpha_beta_ratio, signal_variance, attention_index

# Loop through tasks
for task in TASKS:
    label = task["label"]
    duration = task["duration"]
    instr = task["instructions"]

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(base_dir, f"{participant}_{department}_{label}_{timestamp_str}.csv")

    print("\n====================================")
    print(f"Task: {label} — {duration}s")
    print("Instructions:", instr)
    print("Starting in 5 seconds...")
    time.sleep(5)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms", "eeg_value",
            "alpha_power", "beta_power", "theta_power",
            "alpha_beta_ratio", "signal_variance", "attention_index"
        ])

        buffer = []
        start_time = time.time()
        while (time.time() - start_time) < duration:
            line = ser.readline().decode('utf-8').strip()
            if not line or "," not in line:
                continue

            try:
                t_ms_str, eeg_str = line.split(",")
                t_ms = int(t_ms_str)
                eeg_val = float(eeg_str)
            except ValueError:
                continue

            buffer.append(eeg_val)

            if len(buffer) >= SAMPLE_RATE * WINDOW_SEC:
                eeg_window = np.array(buffer[-SAMPLE_RATE * WINDOW_SEC:])
                alpha, beta, theta, abr, var, attn = compute_features(eeg_window)
                writer.writerow([t_ms, eeg_val, alpha, beta, theta, abr, var, attn])

    print(f"Saved {filename}")

print("\nAll tasks completed. Data saved in:", base_dir)
ser.close()


