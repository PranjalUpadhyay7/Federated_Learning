import torch
import torch.nn as nn
from flwr.client import ClientApp
from flwr.common import Message, Context, RecordDict, ArrayRecord, MetricRecord
from centralized.model import TimeSeriesLSTM
from federated.utils import load_partition
from common.dataset import TARGET_CONFIG

# --- Device Selection Utility ---
def select_device(partition_id: int):
    """
    Assigns a device (CPU or GPU) based on the client's partition ID.
    Always falls back to CPU if no CUDA devices are available.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        # Assign GPU based on client ID modulo the number of available GPUs
        device_id = partition_id % num_gpus
        device = torch.device(f"cuda:{device_id}")
        # print(f"Client {partition_id}: Assigned to {device}") # Logging can flood the console
        return device
    
    device = torch.device("cpu")
    # print(f"Client {partition_id}: Assigned to {device} (No CUDA detected)")
    return device

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    device = select_device(partition_id)
    
    # --- READ CONFIG FROM SERVER MESSAGE ---
    local_epochs = msg.content["config"]["num-local-epochs"]
    lr = msg.content["config"]["lr"]
    batch_size = msg.content["config"]["batch-size"]
    
    # Load data using the dynamic batch size
    train_loader, _ = load_partition(partition_id, batch_size=batch_size)
    
    model = TimeSeriesLSTM()
    
    if "arrays" in msg.content:
        state_dict = msg.content["arrays"].to_torch_state_dict()
        model.load_state_dict(state_dict)
    
    model.to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    total_loss = 0.0
    
    if len(train_loader) > 0:
        for epoch in range(local_epochs):
            for X_batch, y_batch in train_loader:
                # Move data batch to the device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                loss = 0
                for i, out in enumerate(outputs):
                    loss += criterion(out, y_batch[:, i])
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    avg_loss = total_loss / (len(train_loader) * local_epochs) if len(train_loader) > 0 and local_epochs > 0 else 0.0

    # Ensure model parameters are back on CPU for communication
    new_weights = ArrayRecord(model.cpu().state_dict())
    metrics = MetricRecord({
        "train_loss": avg_loss,
        "num-examples": len(train_loader.dataset) if len(train_loader) > 0 else 0
    })
    
    return Message(content=RecordDict({"arrays": new_weights, "metrics": metrics}), reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    device = select_device(partition_id)
    
    # Use a fixed batch size for evaluation if not specified in config
    eval_batch_size = 64
    _, test_loader = load_partition(partition_id, batch_size=eval_batch_size)
    
    model = TimeSeriesLSTM()
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)
    model.to(device) # Move model to device
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        if len(test_loader) > 0:
            for X_batch, y_batch in test_loader:
                # Move data batch to the device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                batch_loss = 0
                for i, out in enumerate(outputs):
                    # Loss
                    batch_loss += criterion(out, y_batch[:, i]).item()
                    
                    # Accuracy (averaged across all tasks for simplicity)
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == y_batch[:, i]).sum().item()
                    total_predictions += y_batch.size(0)
                
                total_loss += batch_loss

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    avg_acc = correct / total_predictions if total_predictions > 0 else 0.0
    
    metrics = MetricRecord({
        "eval_loss": avg_loss, 
        "accuracy": avg_acc,
        "num-examples": len(test_loader.dataset) if len(test_loader) > 0 else 0
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)