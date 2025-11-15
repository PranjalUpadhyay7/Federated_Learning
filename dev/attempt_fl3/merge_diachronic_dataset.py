# =======================================
# merge_diachronic_dataset.py (v2)
# =======================================
import pandas as pd
from config import TIMEDIARIES_PARQUET, OUTPUT_DIR
from utils import read_parquet_folder, save_csv

def merge_behavior_gt():
    print("🔹 Loading behavioral (v2) and GT data...")

    sensors = pd.read_csv(f"{OUTPUT_DIR}/sensor_features_halfhour_v2.csv")
    sensors["bin"] = pd.to_datetime(sensors["bin"])

    diary = read_parquet_folder(TIMEDIARIES_PARQUET)
    diary["instancetimestamp"] = pd.to_datetime(
        diary["instancetimestamp"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    diary = diary.dropna(subset=["instancetimestamp"])
    diary["bin"] = diary["instancetimestamp"].dt.floor("30min")

    gt_cols = ["userid", "bin", "A4", "A5", "A6a"]
    diary = diary[gt_cols]

    df = sensors.merge(diary, on=["userid", "bin"], how="inner")
    df = df.fillna(0)

    save_csv(df, f"{OUTPUT_DIR}/final_dataset_halfhour_v2.csv")
    print(f"✅ Final dataset shape: {df.shape}")

if __name__ == "__main__":
    merge_behavior_gt()
