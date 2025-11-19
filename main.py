import torch
import numpy as np
import sys
import os
from sklearn.metrics import classification_report

# Ensures the project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from centralized.model import TimeSeriesLSTM, StaticNet
from common.utils import load_model_weights
from common.dataset import load_all_centralized, TARGET_CONFIG

def evaluate_on_test_set(model, test_loader):
    """Evaluates the model on the full test set and returns predictions and truths."""
    model.eval()
    all_preds = [[] for _ in range(len(TARGET_CONFIG))]
    all_truths = [[] for _ in range(len(TARGET_CONFIG))]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            for i, out in enumerate(outputs):
                # Get predicted class index
                preds = torch.argmax(out, dim=1).cpu().numpy()
                truths = y_batch[:, i].cpu().numpy()
                
                all_preds[i].extend(preds)
                all_truths[i].extend(truths)
                
    return all_preds, all_truths

def main():
    print(f"=== Model Comparison Evaluation (Config: {TARGET_CONFIG}) ===")
    
    # 1. Load Data: Get the centralized test set (Ground Truth data)
    # We load the data twice for the two different input shapes required by the models
    _, lstm_test_loader = load_all_centralized(is_static=False)
    _, static_test_loader = load_all_centralized(is_static=True)

    if lstm_test_loader is None or static_test_loader is None:
        print("Error: Could not load centralized test data for evaluation.")
        return
    
    print(f"Loaded {len(lstm_test_loader.dataset)} test samples for comparison.")
    
    # 2. Initialize and Load Models
    models = {
        "LSTM_Central": TimeSeriesLSTM(),
        "LSTM_Federated": TimeSeriesLSTM(),
        "StaticNet_Central": StaticNet(),
        "StaticNet_Federated": StaticNet(),
    }

    # Load Weights
    models["LSTM_Central"] = load_model_weights(models["LSTM_Central"], "saved_models/centralized_lstm.pth")
    models["LSTM_Federated"] = load_model_weights(models["LSTM_Federated"], "saved_models/federated_lstm.pth")
    models["StaticNet_Central"] = load_model_weights(models["StaticNet_Central"], "saved_models/centralized_static.pth")
    # Check for Federated Static weights before loading
    if "federated_static.pth" in os.listdir("saved_models"):
        models["StaticNet_Federated"] = load_model_weights(models["StaticNet_Federated"], "saved_models/federated_static.pth") 
    else:
        print("Warning: Federated StaticNet model not found. Skipping its evaluation.")

    # 3. Evaluate Models
    results = {}
    truths = {}
    
    # LSTM Models use LSTM test loader
    lstm_models = ["LSTM_Central", "LSTM_Federated"]
    for name in lstm_models:
        results[name], truths_lstm = evaluate_on_test_set(models[name], lstm_test_loader)
        
    # StaticNet Models use Static test loader
    static_models = ["StaticNet_Central", "StaticNet_Federated"]
    for name in static_models:
        if "StaticNet_Federated" in name and "federated_static.pth" not in os.listdir("saved_models"):
            continue
        results[name], truths_static = evaluate_on_test_set(models[name], static_test_loader)
        
    # 4. Generate Classification Reports
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT COMPARISON (Evaluated on Combined Test Set)")
    print("="*80)
    
    for i, num_classes in enumerate(TARGET_CONFIG):
        print(f"\n--- Task {i} (Classes: 0 to {num_classes-1}) ---")
        
        target_names = [f'Class {c}' for c in range(num_classes)]
        
        for name in list(models.keys()):
            if "StaticNet_Federated" in name and "federated_static.pth" not in os.listdir("saved_models"):
                continue 
            
            # Select appropriate ground truth
            current_truths = truths_lstm if "LSTM" in name else truths_static
                
            print(f"\n[{name.upper()} REPORT]")
            print(classification_report(current_truths[i], results[name][i], 
                                        zero_division=0, 
                                        target_names=target_names))

if __name__ == "__main__":
    main()