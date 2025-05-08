import sys
import os

project_root = '/Users/irwin/Documents/GitHub/lisa-ml'
sys.path.append(project_root)
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import ndf


class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, num_model_classes):
        super(MultiTaskClassifier, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Binary head: outputs continuous probability using Sigmoid activation.
        self.binary_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        # Model head: outputs raw logits for multi-class classification.
        self.model_head = nn.Linear(32, num_model_classes)

    def forward(self, x):
        shared_out = self.shared(x)
        binary_out = self.binary_head(shared_out)
        model_out = self.model_head(shared_out)
        return binary_out, model_out


class MultiTaskClassifier_L(nn.Module):
    def __init__(self, input_dim, num_model_classes):
        super(MultiTaskClassifier_L, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        # Multi-class classification head
        self.model_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_model_classes),
        )

    def forward(self, x):
        shared_out = self.shared(x)
        binary_out = self.binary_head(shared_out)
        model_out = self.model_head(shared_out)
        return binary_out, model_out


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid(),   # outputs in [0,1]
        )

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        returns: [batch_size, 1] (probability of class=1)
        """
        shared_out = self.shared(x)
        binary_out = self.binary_head(shared_out)
        return binary_out

class BinaryFeatureLayer(nn.Module):
    """
    Wraps exactly the shared trunk from your BinaryClassifierL, and
    tells the forest that its output dimension is 32.
    """
    def __init__(self, input_dim):
        super().__init__()
        # copy‐paste your shared definition here:
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(), nn.Dropout(0.3),
        )
        # hard‐code the output dim of that trunk:
        self._out_dim = 32

    def forward(self, x):
        # produces [batch,32]
        return self.shared(x)

    def get_out_feature_size(self):
        return self._out_dim
    
def prepare_ndf_model(opt, input_dim):
    feat_layer = BinaryFeatureLayer(input_dim)

    forest = ndf.Forest(
        n_tree            = opt['n_tree'],
        tree_depth        = opt['tree_depth'],
        n_in_feature      = feat_layer.get_out_feature_size(),
        tree_feature_rate = opt['tree_feature_rate'],
        n_class           = 2,                      
        jointly_training  = opt['jointly_training'],
    )

    model = ndf.NeuralDecisionForest(feat_layer, forest)
    return model.cpu() 