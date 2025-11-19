import pandas as pd
import os
import shutil

def split_csv_per_client(input_filepath, output_folder, features_to_keep=None):
    """
    Reads a large CSV with headers, groups by 'userid', 
    and saves individual CSVs (without headers) for the FL pipeline.
    
    Args:
        input_filepath (str): Path to the source CSV.
        output_folder (str): Directory to save user files.
        features_to_keep (list): List of column names to retain in the output. 
                                 If None (default), ALL columns are kept.
    """
    
    # 1. Setup Output Directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        # Optional: Clear folder to ensure no old data remains
        print(f"Output folder '{output_folder}' exists. Cleaning up...")
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    print(f"Reading {input_filepath}...")
    
    try:
        # 2. Read the source CSV
        df = pd.read_csv(input_filepath)
        
        # 3. Verify 'userid' exists (needed for splitting)
        if 'userid' not in df.columns:
            print("Error: Column 'userid' not found in the CSV.")
            return

        # 4. Group by User ID
        unique_users = df['userid'].unique()
        print(f"Found {len(unique_users)} unique clients. Splitting files...")

        for user_id in unique_users:
            # Filter data for this specific user
            client_data = df[df['userid'] == user_id]
            
            # 5. Filter Columns (Features to Keep)
            # Only apply filter if a specific list is provided
            if features_to_keep is not None:
                # Check if all requested features exist
                available_features = [f for f in features_to_keep if f in client_data.columns]
                
                if len(available_features) != len(features_to_keep):
                    missing = set(features_to_keep) - set(available_features)
                    print(f"Warning: Missing columns in data: {missing}")
                
                # Keep only the requested columns
                client_data = client_data[available_features]

            # Define filename (e.g., real_data/user_A6a.csv)
            safe_filename = f"user_{str(user_id)}.csv"
            output_path = os.path.join(output_folder, safe_filename)
            
            # 6. Save to CSV
            # header=False: Important for your common/dataset.py loader
            # index=False: Don't save the pandas row index number
            client_data.to_csv(output_path, index=False, header=False)
            
        print(f"Successfully created {len(unique_users)} files in '{output_folder}/'")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "new.csv"
    OUTPUT_DIR = "real_data_v2"
    
    # LIST OF FEATURES TO KEEP
    # Set to None to keep ALL columns by default.
    # Or provide a list of strings: ["col1", "col2", "target"]
    FEATURES = None 
    
    # Example of specific selection (Uncomment to use):
    features = [
    "app_event_count",
    "unique_apps_count",
    "app_usage_entropy",
    "music_play_count",
    "headset_plug_count",
    "notif_posted_count",
    "notif_unique_packages",
    "notif_entropy",
    "airplane_on_count",
    "battery_charge_on_count",
    "battery_level_mean",
    "doze_on_count",
    "ring_mode_normal_count",
    "ring_mode_silent_count",
    "ring_mode_vibrate_count",
    "screen_on_count",
    "touch_count",
    "user_present_count",
    "motion_invehicle_count",
    "motion_onbicycle_count",
    "motion_onfoot_count",
    "motion_still_count",
    "motion_unknown_count",
    "motion_walking_count",
    "motion_tilting_count",
    "step_count",
    "A6a",
]


    
    # Run function
    split_csv_per_client(INPUT_FILE, OUTPUT_DIR, features_to_keep=features)