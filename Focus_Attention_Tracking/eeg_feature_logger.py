import serial
import csv
import numpy as np
from datetime import datetime
from scipy.signal import welch

# -------- SETTINGS --------
PORT = "COM6"       # Change if needed
BAUD = 115200
SAMPLE_RATE = 250   # Hz (matches Arduino delay ~4ms)
WINDOW_SEC = 2      # For feature computation
# --------------------------

# Ask for participant & activity
participant = input("Enter participant name: ").strip()
activity = input("Enter activity type (baseline/focus1/distraction/focus2): ").strip().lower()

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
FILENAME = f"{participant}_{activity}_{timestamp_str}.csv"

# Connect to EEG device
ser = serial.Serial(PORT, BAUD)
print(f"Connected to {PORT} at {BAUD} baud")
print(f"Saving to {FILENAME}")

# Open CSV for writing
with open(FILENAME, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_ms", "eeg_value",
        "alpha_power", "beta_power", "theta_power",
        "alpha_beta_ratio", "signal_variance",
        "attention_index"
    ])

    buffer = []
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line or "," not in line:
                continue

            t_ms_str, eeg_str = line.split(",")
            try:
                t_ms = int(t_ms_str)
                eeg_val = float(eeg_str)
            except ValueError:
                continue

            buffer.append(eeg_val)

            # If enough samples for a window
            if len(buffer) >= SAMPLE_RATE * WINDOW_SEC:
                eeg_window = np.array(buffer[-SAMPLE_RATE * WINDOW_SEC:])

                # Compute bandpowers
                freqs, psd = welch(eeg_window, fs=SAMPLE_RATE, nperseg=256)
                alpha_mask = (freqs >= 8) & (freqs <= 12)
                beta_mask  = (freqs >= 13) & (freqs <= 30)
                theta_mask = (freqs >= 4) & (freqs <= 7)
                alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
                beta_power  = np.trapezoid(psd[beta_mask], freqs[beta_mask])
                theta_power = np.trapezoid(psd[theta_mask], freqs[theta_mask])


                alpha_beta_ratio = alpha_power / beta_power if beta_power != 0 else 0
                signal_variance = np.var(eeg_window)

                # Simple attention index (customizable)
                attention_index = beta_power / (alpha_power + theta_power + 1e-6)

                # Save row1
                writer.writerow([
                    t_ms, eeg_val,
                    alpha_power, beta_power, theta_power,
                    alpha_beta_ratio, signal_variance,
                    attention_index
                ])

    except KeyboardInterrupt:
        print("\nData collection stopped.")
        ser.close()
