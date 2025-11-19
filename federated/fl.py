import flwr as fl
import torch
import os
import sys
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg

# Ensure project root is in path to access modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from centralized.model import TimeSeriesLSTM, StaticNet
from federated.utils import load_partition
from common.dataset import TARGET_CONFIG, get_files_sorted # Import file counter

# --- DYNAMIC CONFIGURATION ---
# 1. Detect actual number of clients from data files
AVAILABLE_FILES = get_files_sorted()
NUM_CLIENTS = len(AVAILABLE_FILES)

if NUM_CLIENTS == 0:
    print("ERROR: No CSV files found in 'users_data/'. Please run 'python common/synthetic_data.py' or add your own data.")
    sys.exit(1)

print(f"--- DYNAMIC SETUP ---")
print(f"Detected {NUM_CLIENTS} unique user files. Setting NUM_CLIENTS to {NUM_CLIENTS}.")

# 2. FL Hyperparameters
NUM_ROUNDS = 50   
FRACTION_FIT = 0.8 
LOCAL_EPOCHS = 5  
LR = 0.0005       
BATCH_SIZE = 64   

# --- CLIENT BASE CLASS ---
class BaseFlowerClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int, model_class, is_static: bool):
        self.partition_id = partition_id
        # Pass is_static flag to data loader
        self.train_loader, self.test_loader = load_partition(partition_id, batch_size=BATCH_SIZE, is_static=is_static)
        self.model = model_class()
        self.device = self._select_device()

    def _select_device(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # Distribute clients across GPUs
            device_id = self.partition_id % num_gpus
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")

    def get_parameters(self, config):
        # Must return CPU numpy arrays for communication
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _train_evaluate_logic(self, parameters):
        self.model.to(self.device)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)
        return torch.nn.CrossEntropyLoss(), torch.optim.Adam(self.model.parameters(), lr=LR)
    
    def fit(self, parameters, config):
        criterion, optimizer = self._train_evaluate_logic(parameters)
        self.model.train()
        
        if len(self.train_loader) > 0:
            for epoch in range(LOCAL_EPOCHS):
                for X_batch, y_batch in self.train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = sum(criterion(out, y_batch[:, i]) for i, out in enumerate(outputs))
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
                loss += sum(criterion(out, y_batch[:, i]) for i, out in enumerate(outputs)).item()
        
        return loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0, \
               len(self.test_loader.dataset), \
               {"accuracy": 0.0}

# --- PARALLEL CLIENT CLASSES ---
class TimeSeriesFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, TimeSeriesLSTM, is_static=False)

class StaticFlowerClient(BaseFlowerClient):
    def __init__(self, partition_id: int):
        super().__init__(partition_id, StaticNet, is_static=True)


# --- 3. CUSTOM STRATEGY TO SAVE MODEL ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_class, save_path, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.save_path = save_path

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            print(f"\nAggregating final model ({self.model_class.__name__}) and saving...")
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            net = self.model_class()
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), ndarrays)})
            net.load_state_dict(state_dict, strict=True)
            
            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")
            torch.save(net.state_dict(), self.save_path)
            print(f"Saved global FL model to {self.save_path}")

        return aggregated_parameters, aggregated_metrics

# --- 4. START SIMULATION FUNCTIONS ---
def run_fl_lstm():
    print("\n" + "="*50)
    print("--- Running FL Simulation (LSTM) ---")
    print("="*50)
    
    # Ensure we don't ask for more clients than exist or 0 clients
    min_clients = max(1, int(NUM_CLIENTS * FRACTION_FIT))
    
    strategy = SaveModelStrategy(
        model_class=TimeSeriesLSTM,
        save_path="saved_models/federated_lstm.pth",
        fraction_fit=FRACTION_FIT, 
        fraction_evaluate=FRACTION_FIT,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=NUM_CLIENTS, # Explicitly set available clients
    )
    
    fl.simulation.start_simulation(
        client_fn=lambda cid: TimeSeriesFlowerClient(int(cid)),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

def run_fl_static():
    print("\n" + "="*50)
    print("--- Running FL Simulation (StaticNet) ---")
    print("="*50)
    
    min_clients = max(1, int(NUM_CLIENTS * FRACTION_FIT))
    
    strategy = SaveModelStrategy(
        model_class=StaticNet,
        save_path="saved_models/federated_static.pth",
        fraction_fit=FRACTION_FIT, 
        fraction_evaluate=FRACTION_FIT,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=NUM_CLIENTS,
    )
    
    fl.simulation.start_simulation(
        client_fn=lambda cid: StaticFlowerClient(int(cid)),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    # Runs LSTM FL simulation, followed by StaticNet FL simulation
    run_fl_lstm()
    run_fl_static()