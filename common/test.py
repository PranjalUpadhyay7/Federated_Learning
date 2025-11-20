import os
import pandas as pd
import numpy as np

# Configuration matching your dataset.py
DATA_DIR = "real_data_v2"
NUM_FEATURES = 26
TARGET_CONFIG = [5] 
NUM_TARGETS = len(TARGET_CONFIG)

def check_data_labels():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"Checking {len(files)} files...")

    all_targets = []

    for f in files:
        filepath = os.path.join(DATA_DIR, f)
        # Load raw data
        try:
            df = pd.read_csv(filepath, header=None)
            data = df.values
            
            # Extract the target columns (Everything after NUM_FEATURES)
            # In your case, column index 5
            targets = data[:, NUM_FEATURES : NUM_FEATURES + NUM_TARGETS]
            all_targets.append(targets)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_targets:
        print("No data found.")
        return

    # Flatten all targets to see global unique values
    all_targets = np.concatenate(all_targets, axis=0)

    print("\n--- DATA REPORT ---")
    for i in range(NUM_TARGETS):
        col_targets = all_targets[:, i]
        
        # Get unique values and their counts
        unique_vals, counts = np.unique(col_targets, return_counts=True)
        
        min_val = np.min(unique_vals)
        max_val = np.max(unique_vals)
        
        print(f"Task {i}:")
        print(f"  Expected Classes: 0 to {TARGET_CONFIG[i] - 1}")
        print(f"  Range Found: [{min_val}, {max_val}]")
        
        print("  Class Distribution:")
        for val, count in zip(unique_vals, counts):
            print(f"    Class {val}: {count} instances")
            
        if min_val < 0 or max_val >= TARGET_CONFIG[i]:
            print(f"  >>> CRITICAL ISSUE: Values outside [0, {TARGET_CONFIG[i]-1}] range!")
            if min_val == 1:
                print("  >>> HINT: Your data looks 1-indexed (starts at 1). You need to subtract 1.")
        else:
            print("  >>> Status: OK")

if __name__ == "__main__":
    check_data_labels()