# =======================================
# config.py
# =======================================
import os

# === Base folders ===
SENSORS_FOLDER = r"E:\trio_grad\2020-DiversityOne-Trento\Sensors\App-usage"
DIACHRONIC_FOLDER = r"E:\trio_grad\2020-DiversityOne-Trento\Diachronic-Interactions"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Sensor subfolders ===
APP_FOLDER = os.path.join(SENSORS_FOLDER, "application.parquet")
HEADSET_FOLDER = os.path.join(SENSORS_FOLDER, "headsetplug.parquet")
MUSIC_FOLDER = os.path.join(SENSORS_FOLDER, "music.parquet")
NOTIF_FOLDER = os.path.join(SENSORS_FOLDER, "notification.parquet")

# === GT Data (Diachronic Interactions) ===
TIMEDIARIES_PARQUET = os.path.join(DIACHRONIC_FOLDER, "timediaries.parquet")

print("✅ Paths configured successfully!")
print("Sensors folder:", SENSORS_FOLDER)
print("Diachronic folder:", DIACHRONIC_FOLDER)
