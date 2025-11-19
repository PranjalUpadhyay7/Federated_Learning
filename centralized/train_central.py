import torch
import torch.nn as nn
import sys
import os

# Ensures the parent directory (project root) is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centralized.model import TimeSeriesLSTM, StaticNet
from common.dataset import load_all_centralized, TARGET_CONFIG
from common.utils import save_model

# --- CENTRALIZED TRAINING CONFIGURATION ---
# Increase this to maximize GPU usage (e.g., 128, 256, 512)
CENTRAL_BATCH_SIZE = 64  #16384
# Set to number of CPU cores for faster data loading
NUM_WORKERS = 16
CENTRAL_EPOCHS = 150

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
            
            loss = 0
            # Calculate loss and metrics for each task
            for i, out in enumerate(outputs):
                task_loss = criterion(out, y_batch[:, i])
                loss += task_loss
                
                # Accuracy calculation
                preds = torch.argmax(out, dim=1)
                total_correct += (preds == y_batch[:, i]).sum().item()
                total_predictions += y_batch.size(0)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Avg Task Acc: {avg_acc:.2%}")

    save_model(model.cpu(), save_path)

def train_lstm_centralized():
    print(f"\n--- Centralized Training (LSTM Sequential: {TARGET_CONFIG}) ---")
    train_loader, _ = load_all_centralized(batch_size=CENTRAL_BATCH_SIZE, num_workers=NUM_WORKERS, is_static=False)
    if train_loader is None: return

    model = TimeSeriesLSTM() 
    run_training_loop(model, train_loader, epochs=CENTRAL_EPOCHS, save_path="saved_models/centralized_lstm.pth")

def train_static_centralized():
    print(f"\n--- Centralized Training (StaticNet MLP: {TARGET_CONFIG}) ---")
    train_loader, _ = load_all_centralized(batch_size=CENTRAL_BATCH_SIZE, num_workers=NUM_WORKERS, is_static=True)
    if train_loader is None: return

    model = StaticNet() 
    run_training_loop(model, train_loader, epochs=CENTRAL_EPOCHS, save_path="saved_models/centralized_static.pth")

if __name__ == "__main__":
    # Run both centralized benchmarks
    train_lstm_centralized()
    train_static_centralized()