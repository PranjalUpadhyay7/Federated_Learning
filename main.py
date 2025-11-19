import torch
import numpy as np
import sys
import os
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from centralized.model import TimeSeriesLSTM_Single, StaticNet_Single, TabM_Single
from common.utils import load_model_weights
from common.dataset import load_all_centralized, TARGET_CONFIG

def evaluate_on_test_set(model, test_loader):
    model.eval()
    all_preds = []
    all_truths = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.squeeze()
            
            # Single output tensor
            outputs = model(X_batch)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            truths = y_batch.cpu().numpy()
            
            all_preds.extend(preds)
            all_truths.extend(truths)
                
    return all_preds, all_truths

def main():
    print(f"=== Model Comparison Evaluation (5 Classes) ===")
    
    # 1. Load Data
    _, lstm_test_loader = load_all_centralized(is_static=False)
    _, static_test_loader = load_all_centralized(is_static=True)

    if lstm_test_loader is None or static_test_loader is None:
        return
    
    # 2. Initialize Models
    models = {
        "LSTM_Central": TimeSeriesLSTM_Single(),
        "LSTM_Federated": TimeSeriesLSTM_Single(),
        "StaticNet_Central": StaticNet_Single(),
        "StaticNet_Federated": StaticNet_Single(),
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
    
    target_names = [f'Class {c}' for c in range(TARGET_CONFIG[0])]
    
    for name in loaded_models:
        print(f"\n[{name.upper()} REPORT]")
        print(classification_report(truths_map[name], results[name], 
                                    zero_division=0, 
                                    target_names=target_names))

if __name__ == "__main__":
    main()