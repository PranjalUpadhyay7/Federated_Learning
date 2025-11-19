import os
import numpy as np
from torch.utils.data import DataLoader
from common.dataset import get_files_sorted, load_data_from_file, TimeSeriesDataset, DATA_DIR
from sklearn.model_selection import train_test_split

def load_partition(partition_id: int, batch_size: int, is_static: bool):
    """
    Loads specific user file based on partition ID and returns DataLoaders 
    with the specified batch size and static flag.
    """
    files = get_files_sorted()
    
    if not files:
        raise FileNotFoundError(f"No CSV files in {DATA_DIR}")

    # Map partition_id to file index 
    file_idx = partition_id % len(files)
    filename = files[file_idx]
    filepath = os.path.join(DATA_DIR, filename)
    
    # Use the unified loader, passing the is_static flag
    X, y = load_data_from_file(filepath, is_static) 
    
    if len(X) == 0:
        # Return empty loaders if file is empty
        return DataLoader([]), DataLoader([])

    # Split local user data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pass is_static flag to the Dataset constructor
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, is_static=is_static), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test, is_static=is_static), batch_size=batch_size)
    
    return train_loader, test_loader