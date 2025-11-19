import torch
import torch.nn as nn
import torch.nn.functional as F
from common.dataset import NUM_FEATURES, SEQ_LEN, TARGET_CONFIG

# --- Time Series Model (LSTM) ---

class TimeSeriesLSTM(nn.Module):
    """
    LSTM-based model for sequence data. Used as the base for the current FL/Central pipelines.
    Input shape: (Batch, SEQ_LEN, NUM_FEATURES)
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=1, target_config=TARGET_CONFIG):
        super(TimeSeriesLSTM, self).__init__()
        self.target_config = target_config
        
        # Shared LSTM Backbone
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Multi-Head Output: One Linear layer per task
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for num_classes in target_config
        ])

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Shared representations
        lstm_out, _ = self.lstm(x)
        # Take last time step: (batch, hidden_size)
        last_step = lstm_out[:, -1, :]
        
        # Forward pass through each head
        outputs = []
        for head in self.heads:
            outputs.append(head(last_step))
            
        return outputs

# --- Static Data Model (Robust Deep MLP) ---

class StaticNet(nn.Module):
    """
    Deep MLP model that processes static feature vectors (non-sequential).
    Input shape: (Batch, NUM_FEATURES)
    """
    def __init__(self, hidden_size=128, target_config=TARGET_CONFIG):
        super(StaticNet, self).__init__()
        
        # CRITICAL CHANGE: Input dimension is just NUM_FEATURES, as SEQ_LEN is ignored.
        INPUT_DIM = NUM_FEATURES 
        
        # Robust Deep Learning Architecture (MLP)
        self.layers = nn.Sequential(
            # First Layer
            nn.Linear(INPUT_DIM, hidden_size),
            nn.BatchNorm1d(hidden_size), # Stabilizes training
            nn.ReLU(),
            nn.Dropout(0.3), # Regularization
            
            # Second Layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Determine the final feature size before the heads
        FINAL_HIDDEN_SIZE = hidden_size // 2
        
        # Multi-Head Output: One Linear layer per task
        self.heads = nn.ModuleList([
            nn.Linear(FINAL_HIDDEN_SIZE, num_classes) for num_classes in target_config
        ])

    def forward(self, x):
        # 1. Input x is now (Batch, NUM_FEATURES). No flattening required.
        
        # 2. Pass through shared deep MLP layers
        shared_features = self.layers(x)
        
        # 3. Forward pass through each classification head
        outputs = []
        for head in self.heads:
            outputs.append(head(shared_features))
            
        return outputs