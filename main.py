import torch
import numpy as np
import sys
import os
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from centralized.model import TimeSeriesLSTM, StaticNet, TabM_Single
from common.utils import load_model_weights
from common.dataset import load_all_centralized, TARGET_CONFIG

def evaluate_on_test_set(model, test_loader):
    model.eval()
    # Lists to hold predictions for each task
    all_preds = [[] for _ in range(len(TARGET_CONFIG))]
    all_truths = [[] for _ in range(len(TARGET_CONFIG))]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            
            # Normalize outputs to be a list for consistent processing
            if not isinstance(outputs, list):
                outputs = [outputs]
                
            # Process each task
            for i, out in enumerate(outputs):
                preds = torch.argmax(out, dim=1).cpu().numpy()
                
                # Handle single vs multi-target y_batch
                if y_batch.dim() > 1 and y_batch.shape[1] > 1:
                    truths = y_batch[:, i].cpu().numpy()
                else:
                    # Single task or squeezed target
                    truths = y_batch.squeeze().cpu().numpy()

                all_preds[i].extend(preds)
                all_truths[i].extend(truths)
                
    return all_preds, all_truths

def main():
    print(f"=== Model Comparison Evaluation (Config: {TARGET_CONFIG}) ===")
    
    # 1. Load Data
    _, lstm_test_loader = load_all_centralized(is_static=False)
    _, static_test_loader = load_all_centralized(is_static=True)

    if lstm_test_loader is None or static_test_loader is None:
        return
    
    # 2. Initialize Models
    models = {
        "LSTM_Central": TimeSeriesLSTM(),
        "LSTM_Federated": TimeSeriesLSTM(),
        "StaticNet_Central": StaticNet(),
        "StaticNet_Federated": StaticNet(),
        "TabM_Central": TabM_Single(),
        "TabM_Federated": TabM_Single(),
    }

    # 3. Load Weights
    model_files = {
        "LSTM_Central": "saved_models/centralized_lstm_v2.pth",
        "LSTM_Federated": "saved_models/federated_lstm_v2.pth",
        "StaticNet_Central": "saved_models/centralized_static_v2.pth",
        "StaticNet_Federated": "saved_models/federated_static_v2.pth",
        "TabM_Central": "saved_models/centralized_tabm_v2.pth",
        "TabM_Federated": "saved_models/federated_tabm_v2.pth",
    }

    loaded_models = []
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = load_model_weights(models[name], path)
            loaded_models.append(name)
        else:
            print(f"Warning: {name} not found at {path}. Skipping.")

    # 4. Evaluate
    results = {}
    truths_map = {}
    
    for name in loaded_models:
        if "LSTM" in name:
            results[name], truths_map[name] = evaluate_on_test_set(models[name], lstm_test_loader)
        else:
            results[name], truths_map[name] = evaluate_on_test_set(models[name], static_test_loader)
        
    # 5. Reports
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT COMPARISON")
    print("="*80)
    
    for i, num_classes in enumerate(TARGET_CONFIG):
        print(f"\n--- Task {i} (Classes: 0 to {num_classes-1}) ---")
        target_names = [f'Class {c}' for c in range(num_classes)]
        
        for name in loaded_models:
            print(f"\n[{name.upper()} REPORT]")
            print(classification_report(truths_map[name][i], results[name][i], 
                                        zero_division=0, 
                                        target_names=target_names))

if __name__ == "__main__":
    main()