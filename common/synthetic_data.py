import os
import numpy as np
import pandas as pd
import random
from dataset import NUM_FEATURES, TARGET_CONFIG, DATA_DIR, NUM_USERS

# Range for random samples per user
MIN_SAMPLES = 300
MAX_SAMPLES = 400

def generate_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Generating data for {NUM_USERS} users...")
    print(f"Features: {NUM_FEATURES}")
    print(f"Targets Configuration: {TARGET_CONFIG} (Classes per task)")
    print(f"Samples per user: Random between {MIN_SAMPLES} and {MAX_SAMPLES}")

    for i in range(1, NUM_USERS + 1):
        # Random sample count for this user
        num_samples = random.randint(MIN_SAMPLES, MAX_SAMPLES)

        # 1. Generate features
        X_data = np.random.randn(num_samples, NUM_FEATURES)

        # 2. Multi-head classification targets
        y_list = []
        for num_classes in TARGET_CONFIG:
            y_col = np.random.randint(0, num_classes, size=(num_samples, 1))
            y_list.append(y_col)

        Y_data = np.hstack(y_list)

        # Combine features + targets
        data = np.hstack((X_data, Y_data))

        # Save CSV
        df = pd.DataFrame(data)
        filename = f"user_{i}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, header=False, index=False)

        print(f"User {i}: {num_samples} samples → saved {filename}")

    print("\nData generation complete.")

if __name__ == "__main__":
    generate_data()
