import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centralized.model import TimeSeriesLSTM, StaticNet, TabM_Single
from common.dataset import load_all_centralized, TARGET_CONFIG
from common.utils import save_model

# --- CONFIGURATION ---
CENTRAL_BATCH_SIZE = 64 
NUM_WORKERS = 4 
CENTRAL_EPOCHS = 100

def calculate_loss(outputs, y_batch, criterion):
    """Dynamic loss calculation for single or multi-head output."""
    # Case 1: Multi-Head (List of tensors)
    if isinstance(outputs, list):
        loss = 0
        for i, out in enumerate(outputs):
            # y_batch is (Batch, NumTargets)
            target = y_batch[:, i] if y_batch.dim() > 1 else y_batch
            loss += criterion(out, target)
        return loss, outputs[0] # Return first head for simple accuracy logging
        
    # Case 2: Single-Head (Single tensor)
    else:
        # Ensure target is (Batch,)
        if y_batch.dim() > 1:
            y_batch = y_batch.squeeze()
        return criterion(outputs, y_batch), outputs

def run_training_loop(model, train_loader, epochs, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    print(f"Starting training on {device} with batch size {CENTRAL_BATCH_SIZE}...")
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Dynamic Loss Calculation
            loss, main_output = calculate_loss(outputs, y_batch, criterion)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Simple accuracy logging (using first task if multi-head)
            if y_batch.dim() > 1: y_batch_main = y_batch[:, 0]
            else: y_batch_main = y_batch
                
            preds = torch.argmax(main_output, dim=1)
            total_correct += (preds == y_batch_main).sum().item()
            total_predictions += y_batch_main.size(0)
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Main Task Acc: {avg_acc:.2%}")

    save_model(model.cpu(), save_path)

def train_lstm_centralized():
    print(f"\n--- Centralized Training (LSTM Generic: {TARGET_CONFIG}) ---")
    train_loader, _ = load_all_centralized(batch_size=CENTRAL_BATCH_SIZE, num_workers=NUM_WORKERS, is_static=False)
    if train_loader is None: return
    model = TimeSeriesLSTM() 
    run_training_loop(model, train_loader, epochs=CENTRAL_EPOCHS, save_path="saved_models/centralized_lstm_v2.pth")

def train_static_centralized():
    print(f"\n--- Centralized Training (StaticNet Generic: {TARGET_CONFIG}) ---")
    train_loader, _ = load_all_centralized(batch_size=CENTRAL_BATCH_SIZE, num_workers=NUM_WORKERS, is_static=True)
    if train_loader is None: return
    model = StaticNet() 
    run_training_loop(model, train_loader, epochs=CENTRAL_EPOCHS, save_path="saved_models/centralized_static_v2.pth")

def train_tabm_centralized():
    print(f"\n--- Centralized Training (TabM Generic: {TARGET_CONFIG}) ---")
    train_loader, _ = load_all_centralized(batch_size=CENTRAL_BATCH_SIZE, num_workers=NUM_WORKERS, is_static=True)
    if train_loader is None: return
    model = TabM_Single() 
    run_training_loop(model, train_loader, epochs=CENTRAL_EPOCHS, save_path="saved_models/centralized_tabm_v2.pth")

if __name__ == "__main__":
    # train_lstm_centralized()
    # train_static_centralized()
    train_tabm_centralized()