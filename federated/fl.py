import flwr as fl
import torch
import os
import sys
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Using Generic models that handle both single and multi-task
from centralized.model import TimeSeriesLSTM, StaticNet, TabM_Single
from federated.utils import load_partition
from common.dataset import TARGET_CONFIG, get_files_sorted 

# --- DYNAMIC CONFIGURATION ---
AVAILABLE_FILES = get_files_sorted()
NUM_CLIENTS = len(AVAILABLE_FILES)

if NUM_CLIENTS == 0:
    print("ERROR: No CSV files found in 'users_data/'.")
    sys.exit(1)

print(f"--- DYNAMIC SETUP ---")
print(f"Detected {NUM_CLIENTS} unique user files. Setting NUM_CLIENTS to {NUM_CLIENTS}.")

NUM_ROUNDS = 50   
FRACTION_FIT = 1.0
LOCAL_EPOCHS = 5
LR = 0.0005       
BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 10 # Save every 10 rounds

# --- CLIENT BASE CLASS ---
class BaseFlowerClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int, model_class, is_static: bool):
        self.partition_id = partition_id
        self.train_loader, self.test_loader = load_partition(partition_id, batch_size=BATCH_SIZE, is_static=is_static)
        self.model = model_class()
        self.device = self._select_device()

    def _select_device(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device_id = self.partition_id % num_gpus
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _train_evaluate_logic(self, parameters):
        self.model.to(self.device)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)
        # Define weights (Ensure they are on the correct device: self.device)
        class_weights = torch.tensor([1.0, 0.36, 0.86, 9.8, 24.0]).to(self.device)
        
        # Use Weighted Loss
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        return criterion, torch.optim.Adam(self.model.parameters(), lr=LR)

    def _calculate_loss(self, outputs, y_batch, criterion):
        """Handles both list (multi-head) and tensor (single-head) outputs."""
        if isinstance(outputs, list):
            loss = 0
            for i, out in enumerate(outputs):
                target = y_batch[:, i] if y_batch.dim() > 1 else y_batch
                loss += criterion(out, target)
            return loss
        else:
            # Ensure target is (Batch,)
            if y_batch.dim() > 1: y_batch = y_batch.squeeze()
            return criterion(outputs, y_batch)
    
    def fit(self, parameters, config):
        criterion, optimizer = self._train_evaluate_logic(parameters)
        self.model.train()
        
        if len(self.train_loader) > 0:
            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in self.train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self._calculate_loss(outputs, y_batch, criterion)
                    loss.backward()
                    optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        criterion, _ = self._train_evaluate_logic(parameters)
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss += self._calculate_loss(outputs, y_batch, criterion).item()
        return loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0, len(self.test_loader.dataset), {"accuracy": 0.0}

# --- CLIENT CLASSES ---
class TimeSeriesFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, TimeSeriesLSTM, is_static=False)

class StaticFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, StaticNet, is_static=True)

class TabMFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, TabM_Single, is_static=True)

# --- SAVE STRATEGY WITH CHECKPOINTS ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_class, save_path, checkpoint_interval, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.save_path = save_path
        self.checkpoint_interval = checkpoint_interval
        # Base name (e.g. "federated_lstm_v2") derived from save_path "saved_models/federated_lstm_v2.pth"
        self.base_name = os.path.splitext(os.path.basename(save_path))[0] 

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Save Checkpoint if interval matches OR if it's the final round
            if server_round % self.checkpoint_interval == 0 or server_round == NUM_ROUNDS:
                print(f"\nAggregating model ({self.model_class.__name__}) for Round {server_round}...")
                
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                net = self.model_class()
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), ndarrays)})
                net.load_state_dict(state_dict, strict=True)
                
                # Create main directory and checkpoints directory
                if not os.path.exists("saved_models"): os.makedirs("saved_models")
                if not os.path.exists("saved_models/checkpoints"): os.makedirs("saved_models/checkpoints")
                
                if server_round == NUM_ROUNDS:
                    # Final save
                    filename = self.save_path
                else:
                    # Checkpoint save
                    filename = f"saved_models/checkpoints/{self.base_name}_round_{server_round}.pth"
                
                torch.save(net.state_dict(), filename)
                print(f"Saved global FL model to {filename}")

        return aggregated_parameters, aggregated_metrics

# --- RUN FUNCTIONS ---
def run_simulation_generic(model_class, client_class, save_name):
    print("\n" + "="*50)
    print(f"--- Running FL Simulation ({model_class.__name__}) ---")
    print("="*50)
    min_clients = max(1, int(NUM_CLIENTS * FRACTION_FIT))
    
    strategy = SaveModelStrategy(
        model_class=model_class,
        save_path=f"saved_models/{save_name}.pth",
        checkpoint_interval=CHECKPOINT_INTERVAL, # Pass config
        fraction_fit=FRACTION_FIT, 
        fraction_evaluate=FRACTION_FIT,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=NUM_CLIENTS,
    )
    
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_class(int(cid)),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    # Using Generic classes and v2 filenames
    run_simulation_generic(TimeSeriesLSTM, TimeSeriesFlowerClient, "federated_lstm_v2")
    run_simulation_generic(StaticNet, StaticFlowerClient, "federated_static_v2")
    run_simulation_generic(TabM_Single, TabMFlowerClient, "federated_tabm_v2")