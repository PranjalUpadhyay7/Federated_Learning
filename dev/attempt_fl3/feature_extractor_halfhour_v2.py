# =======================================
# feature_extractor_halfhour_v2.py
# =======================================
import pandas as pd
import numpy as np
from config import APP_FOLDER, MUSIC_FOLDER, HEADSET_FOLDER, NOTIF_FOLDER, OUTPUT_DIR
from utils import read_parquet_folder, save_csv, compute_entropy

# -------------------------------
# Helper: floor to 30min bins
# -------------------------------
def make_bin(df, time_col="timestamp"):
    df[time_col] = pd.to_datetime(df[time_col])
    df["bin"] = df[time_col].dt.floor("30min")
    return df

# -------------------------------
# Application features
# -------------------------------
def extract_application_features():
    df = read_parquet_folder(APP_FOLDER)
    df = make_bin(df)
    grouped = df.groupby(["userid", "bin"])
    feat = grouped.agg(
        app_event_count=("timestamp", "count"),
        unique_apps_count=("applicationname", pd.Series.nunique),
    ).reset_index()

    # --- Entropy of app usage per 30-min window ---
    entropy_list = (
        grouped["applicationname"].apply(compute_entropy)
        .reset_index(name="app_usage_entropy")
    )
    feat = feat.merge(entropy_list, on=["userid", "bin"], how="left")

    # --- Variability feature: rolling std over past 2 hours ---
    feat["app_event_var_2h"] = (
        feat.groupby("userid")["app_event_count"]
        .rolling(4, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    return feat.fillna(0)

# -------------------------------
# Music features
# -------------------------------
def extract_music_features():
    df = read_parquet_folder(MUSIC_FOLDER)
    df = make_bin(df)
    feat = (
        df.groupby(["userid", "bin"])
        .agg(music_play_count=("timestamp", "count"))
        .reset_index()
    )
    # Rolling mean for smoothness
    feat["music_play_mean_2h"] = (
        feat.groupby("userid")["music_play_count"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return feat.fillna(0)

# -------------------------------
# Headset plug features
# -------------------------------
def extract_headset_features():
    df = read_parquet_folder(HEADSET_FOLDER)
    df = make_bin(df)
    feat = (
        df.groupby(["userid", "bin"])
        .agg(headset_plug_count=("timestamp", "count"))
        .reset_index()
    )
    feat["headset_plug_var_2h"] = (
        feat.groupby("userid")["headset_plug_count"]
        .rolling(4, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    return feat.fillna(0)

# -------------------------------
# Notification features
# -------------------------------
def extract_notification_features():
    df = read_parquet_folder(NOTIF_FOLDER)
    df = make_bin(df)
    grouped = df.groupby(["userid", "bin"])
    feat = grouped.agg(
        notif_posted_count=("timestamp", "count"),
        notif_unique_packages=("package", pd.Series.nunique),
    ).reset_index()

    # Notification entropy
    notif_entropy = grouped["package"].apply(compute_entropy).reset_index(name="notif_entropy")
    feat = feat.merge(notif_entropy, on=["userid", "bin"], how="left")

    return feat.fillna(0)

# -------------------------------
# Combine all sensor features
# -------------------------------
def combine_features():
    print("🔹 Extracting advanced behavioral features...")

    app = extract_application_features()
    music = extract_music_features()
    headset = extract_headset_features()
    notif = extract_notification_features()

    df = app.merge(music, on=["userid", "bin"], how="outer")
    df = df.merge(headset, on=["userid", "bin"], how="outer")
    df = df.merge(notif, on=["userid", "bin"], how="outer")
    df = df.fillna(0)

    # --- Derived features ---
    df["day_hour"] = pd.to_datetime(df["bin"]).dt.hour
    df["is_daytime"] = df["day_hour"].between(8, 20).astype(int)
    df["day_night_ratio"] = (
        df.groupby("userid")["is_daytime"].transform("mean")
    )

    save_csv(df, f"{OUTPUT_DIR}/sensor_features_halfhour_v2.csv")
    print(f"✅ Final feature table shape: {df.shape}")

if __name__ == "__main__":
    combine_features()
