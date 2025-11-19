import torch
import torch.nn as nn
import torch.nn.functional as F
from tabm import TabM, EnsembleView, make_tabm_backbone, LinearEnsemble
from rtdl_num_embeddings import LinearReLUEmbeddings
from common.dataset import NUM_FEATURES, SEQ_LEN, TARGET_CONFIG

# ==========================================
#       EXISTING MULTI-HEAD MODELS
# ==========================================

class TimeSeriesLSTM(nn.Module):
    """
    Multi-head LSTM for sequence data.
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=1, target_config=TARGET_CONFIG):
        super(TimeSeriesLSTM, self).__init__()
        self.target_config = target_config
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for num_classes in target_config
        ])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        outputs = []
        for head in self.heads:
            outputs.append(head(last_step))
        return outputs

class StaticNet(nn.Module):
    """
    Multi-head Deep MLP for static data.
    """
    def __init__(self, hidden_size=128, target_config=TARGET_CONFIG):
        super(StaticNet, self).__init__()
        INPUT_DIM = NUM_FEATURES 
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        FINAL_HIDDEN_SIZE = hidden_size // 2
        self.heads = nn.ModuleList([
            nn.Linear(FINAL_HIDDEN_SIZE, num_classes) for num_classes in target_config
        ])

    def forward(self, x):
        shared_features = self.layers(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(shared_features))
        return outputs

class TabMNet(nn.Module):
    """
    Multi-head wrapper around TabM backbone.
    """
    def __init__(self, hidden_size=64, target_config=TARGET_CONFIG):
        super(TabMNet, self).__init__()
        # Using implicit defaults for embeddings here as per TabM.make usage
        embeddings = LinearReLUEmbeddings(NUM_FEATURES, d_embedding=8)
        self.backbone = TabM.make(
            n_num_features=NUM_FEATURES,
            num_embeddings=embeddings,
            d_out=hidden_size, 
            n_layers=3,
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for num_classes in target_config
        ])

    def forward(self, x):
        # TabM output: (Batch, k, d_out)
        features_ensemble = self.backbone(x)
        # Mean pooling over ensemble dimension k
        shared_features = features_ensemble.mean(dim=1)
        outputs = []
        for head in self.heads:
            outputs.append(head(shared_features))
        return outputs


# ==========================================
#       NEW SINGLE-OUTPUT MODELS
# ==========================================

class TimeSeriesLSTM_Single(nn.Module):
    """
    Single-head LSTM for sequence data.
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=1, num_classes=TARGET_CONFIG[0]):
        super(TimeSeriesLSTM_Single, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (Batch, SEQ_LEN, NUM_FEATURES)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        # Output: (Batch, num_classes)
        return self.fc(last_step)

class StaticNet_Single(nn.Module):
    """
    Single-head Deep MLP for static data.
    """
    def __init__(self, hidden_size=128, num_classes=TARGET_CONFIG[0]):
        super(StaticNet_Single, self).__init__()
        INPUT_DIM = NUM_FEATURES 
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        FINAL_HIDDEN_SIZE = hidden_size // 2
        self.fc = nn.Linear(FINAL_HIDDEN_SIZE, num_classes)

    def forward(self, x):
        # x: (Batch, NUM_FEATURES)
        features = self.layers(x)
        # Output: (Batch, num_classes)
        return self.fc(features)

class TabM_Single(nn.Module):
    """
    Single-head TabM implementation using the robust TabM.make factory.
    """
    def __init__(self, num_classes=TARGET_CONFIG[0]):
        super(TabM_Single, self).__init__()
        
        # 1. Define Feature Embeddings
        # TabM requires embeddings for numerical features for best performance.
        d_embedding = 8
        embeddings = LinearReLUEmbeddings(NUM_FEATURES, d_embedding)
        
        # 2. Create Model using TabM.make (The "original code" approach)
        # This handles the EnsembleView and Backbone creation internally.
        self.tabm = TabM.make(
            n_num_features=NUM_FEATURES,
            num_embeddings=embeddings,
            d_out=num_classes,
            # n_layers=3,
        )

    def forward(self, x):
        # x: (Batch, NUM_FEATURES)
        
        # TabM output shape: (Batch, k, d_out)
        # k is the ensemble size (default 32)
        out = self.tabm(x)
        
        # Aggregate (Mean) over the ensemble dimension 'k' (dim 1)
        # to produce a single prediction vector per sample.
        # Final Output: (Batch, num_classes)
        return out.mean(dim=1)