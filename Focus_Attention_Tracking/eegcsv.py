import serial
import csv
from datetime import datetime

# Settings
PORT = "COM6"      # Change if needed
BAUD = 115200
FILENAME = f"eeg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Connect to ESP32
ser = serial.Serial(PORT, BAUD)
print(f"Connected to {PORT} at {BAUD} baud")
print(f"Saving to {FILENAME}")

with open(FILENAME, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms", "eeg_value"])  # CSV header

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line and "," in line:
                writer.writerow(line.split(","))
    except KeyboardInterrupt:
        print("\nData collection stopped.")
        ser.close()
