import torch.nn as nn

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, num_model_classes):
        super(MultiTaskClassifier, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Binary head: outputs continuous probability using Sigmoid activation.
        self.binary_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
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
            nn.Dropout(0.3)
        )
        # Binary classification head with an extra hidden layer.
        self.binary_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        # Multi-class classification head with an extra hidden layer.
        self.model_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_model_classes)
        )
    
    def forward(self, x):
        shared_out = self.shared(x)
        binary_out = self.binary_head(shared_out)
        model_out = self.model_head(shared_out)
        return binary_out, model_out