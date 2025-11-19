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

from centralized.model import TimeSeriesLSTM_Generic, StaticNet_Generic, TabM_Generic
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
FRACTION_FIT = 0.8 
LOCAL_EPOCHS = 1  
LR = 0.0005       
BATCH_SIZE = 64   

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
        return torch.nn.CrossEntropyLoss(), torch.optim.Adam(self.model.parameters(), lr=LR)

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
        
        return loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0, \
               len(self.test_loader.dataset), \
               {"accuracy": 0.0}

# --- CLIENT CLASSES ---
class TimeSeriesFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, TimeSeriesLSTM_Generic, is_static=False)

class StaticFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, StaticNet_Generic, is_static=True)

class TabMFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, TabM_Generic, is_static=True)

# --- SAVE STRATEGY ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_class, save_path, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.save_path = save_path

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            print(f"\nAggregating final model ({self.model_class.__name__}) and saving...")
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            net = self.model_class()
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), ndarrays)})
            net.load_state_dict(state_dict, strict=True)
            if not os.path.exists("saved_models"): os.makedirs("saved_models")
            torch.save(net.state_dict(), self.save_path)
            print(f"Saved global FL model to {self.save_path}")
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
    run_simulation_generic(TimeSeriesLSTM_Generic, TimeSeriesFlowerClient, "federated_lstm_v2")
    run_simulation_generic(StaticNet_Generic, StaticFlowerClient, "federated_static_v2")
    run_simulation_generic(TabM_Generic, TabMFlowerClient, "federated_tabm_v2")