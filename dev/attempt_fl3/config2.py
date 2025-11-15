# config.py

import os

# Folder containing one CSV per user
FEDERATED_USER_FOLDER = r"processed_data/federated_users_halfhour_v2"

# Number of FL rounds
NUM_ROUNDS = 15

# Clients per round (10 randomly sampled of total clients)
CLIENTS_PER_ROUND = 10

# Training hyperparameters (local)
LOCAL_EPOCHS = 5
LOCAL_BATCH_SIZE = 32
LOCAL_LR = 1e-3
