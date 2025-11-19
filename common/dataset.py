import torch
import numpy as np
import pandas as pd
import os
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- GLOBAL DATA CONFIGURATION (Source of Truth) ---
# X dimension (number of features per time step)
NUM_FEATURES = 30  
# Y dimension (classes per classification task)
# Example: [2, 4] -> Task 1: Binary (2 classes); Task 2: 4-way classification.
TARGET_CONFIG = [5] 

# Derived Constants
NUM_TARGETS = len(TARGET_CONFIG)
SEQ_LEN = 10       
DATA_DIR = "real_data"
NUM_USERS = 100 

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, is_static=False):
        # X shape will be either (N, SEQ_LEN, NUM_FEATURES) or (N, NUM_FEATURES)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y-1, dtype=torch.long)
        self.is_static = is_static

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is already formatted for the model type, so just return the item.
        return self.X[idx], self.y[idx]

def get_files_sorted():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    # Sort numerically based on the user ID number
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    return files

def load_raw_data_to_xy(filepath):
    """Loads raw CSV data and returns Numpy array."""
    if not os.path.exists(filepath):
        return np.array([]), np.array([])
    
    # header=None ensures no row is skipped for headers
    df = pd.read_csv(filepath, header=None)
    data = df.values.astype(np.float32)
    return data

def load_data_from_file(filepath, is_static):
    """Loads a single CSV and returns (X, Y) suitable for the model type."""
    data = load_raw_data_to_xy(filepath)
    if data.size == 0:
        return np.array([]), np.array([])
    
    expected_cols = NUM_FEATURES + NUM_TARGETS
    if data.shape[1] < expected_cols:
         # Simplified error handling for simulation
         return np.empty((0,)), np.empty((0,))

    if not is_static:
        # Time Series: Create sequences (X shape: N, SEQ_LEN, NUM_FEATURES)
        return create_sequences(data, SEQ_LEN)
    else:
        # Static: Use raw rows (X shape: N, NUM_FEATURES). No sequencing.
        X = data[:, 0:NUM_FEATURES]
        Y = data[:, NUM_FEATURES : NUM_FEATURES + NUM_TARGETS]
        return X, Y

def create_sequences(data_array, seq_len):
    """Converts raw data into sequences."""
    X, y = [], []
    num_samples = len(data_array) - seq_len
    
    if num_samples <= 0:
        return np.empty((0, SEQ_LEN, NUM_FEATURES)), np.empty((0, NUM_TARGETS))

    for i in range(num_samples):
        seq_x = data_array[i : i + seq_len, 0:NUM_FEATURES] 
        seq_y = data_array[i + seq_len, NUM_FEATURES : NUM_FEATURES + NUM_TARGETS]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)


def load_all_centralized(batch_size=64, num_workers=0, is_static=False):
    """Loads and concatenates data from all user CSV files."""
    files = get_files_sorted()
    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        return None, None

    all_X, all_y = [], []
    for f in files:
        # Use the unified loader, passing the is_static flag
        X, y = load_data_from_file(os.path.join(DATA_DIR, f), is_static)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print("No valid data found in any file.")
        return None, None

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    
    # Using customizable batch_size and num_workers
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, is_static=is_static), 
                              batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test, is_static=is_static), 
                             batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, test_loader