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