import torch
import os

def save_model(model, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model_weights(model, path):
    # Determine device for loading (load on CPU if CUDA not available)
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=map_location))
            print(f"Weights loaded from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"No weights file found at {path}")
    return model