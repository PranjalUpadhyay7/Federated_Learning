# fl_simulation.py

import os
import flwr as fl
import pandas as pd
from flwr.simulation import start_simulation

# IMPORT UPDATED CONFIG FILE
from config2 import (
    FEDERATED_USER_FOLDER,
    NUM_ROUNDS,
    CLIENTS_PER_ROUND,
    LOCAL_EPOCHS,
    LOCAL_BATCH_SIZE,
    LOCAL_LR,
)

from flwr_client_sim import FlowerSimClient


# ----------- DISCOVER ALL USER CSV FILES -----------
def load_user_files():
    user_files = []
    for f in os.listdir(FEDERATED_USER_FOLDER):
        if f.endswith(".csv"):
            user_files.append(os.path.join(FEDERATED_USER_FOLDER, f))

    return sorted(user_files)


# ----------- FLOWER SIMULATION CLIENT FACTORY -----------
def client_fn(user_csv: str):
    df = pd.read_csv(user_csv)

    target_cols = ["A4", "A5", "A6a"]
    X_cols = [c for c in df.columns if c not in target_cols + ["userid", "bin", "token"]]
    input_dim = len(X_cols)

    return FlowerSimClient(
        user_csv=user_csv,
        input_dim=input_dim,
        local_epochs=LOCAL_EPOCHS,
        batch_size=LOCAL_BATCH_SIZE,
        lr=LOCAL_LR,
    )


# -------------- MAIN SIMULATION ENTRY --------------
if __name__ == "__main__":
    user_files = load_user_files()
    total_clients = len(user_files)

    print(f"📁 Loaded {total_clients} simulated clients.")
    print(f"➡ Sampling {CLIENTS_PER_ROUND} clients per round.")
    print(f"➡ Running {NUM_ROUNDS} rounds...\n")

    # FedAvg Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CLIENTS_PER_ROUND / total_clients,
        fraction_evaluate=CLIENTS_PER_ROUND / total_clients,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_evaluate_clients=CLIENTS_PER_ROUND,
        min_available_clients=total_clients,
    )

    # Run Simulation
    start_simulation(
        client_fn=lambda cid: client_fn(user_files[int(cid)]),
        num_clients=total_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\n🎉 Federated Learning Simulation Completed!")
