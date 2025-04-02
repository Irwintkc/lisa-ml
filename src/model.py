import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Binary classification head
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
        # Multi-class classification head
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
    
    
class NeuralDecisionForest(nn.Module):
    def __init__(self, input_dim, num_classes, num_trees=5, tree_depth=3):
        """
        A differentiable decision forest layer.
        
        Args:
          input_dim (int): Dimension of the input features.
          num_classes (int): Number of output classes.
          num_trees (int): Number of trees in the forest.
          tree_depth (int): Depth of each tree.
        """
        super(NeuralDecisionForest, self).__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_leaf_nodes = 2 ** tree_depth
        self.num_decision_nodes = self.num_leaf_nodes - 1

        # A linear layer to compute decision logits for all trees at once.
        # Output shape: (batch_size, num_trees * num_decision_nodes)
        self.decision_layer = nn.Linear(input_dim, num_trees * self.num_decision_nodes)

        # Each tree has its own set of leaf distributions (for class predictions)
        # Shape: (num_trees, num_leaf_nodes, num_classes)
        self.leaf_distributions = nn.Parameter(torch.randn(num_trees, self.num_leaf_nodes, num_classes))
        
        # Precompute the routing matrix for a tree given its depth.
        # This matrix, M, has shape (num_leaf_nodes, num_decision_nodes)
        # and encodes the binary routing decisions along the path from root to each leaf.
        self.register_buffer("routing_matrix", self._get_routing_matrix(tree_depth))
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        # Compute decision logits and reshape for trees.
        decision_logits = self.decision_layer(x)  # (batch_size, num_trees * num_decision_nodes)
        decision_logits = decision_logits.view(batch_size, self.num_trees, self.num_decision_nodes)
        # Convert logits to probabilities using sigmoid.
        decision_probs = torch.sigmoid(decision_logits)  # (batch_size, num_trees, num_decision_nodes)
        
        # Expand dimensions to prepare for routing computation.
        # routing_matrix: (num_leaf_nodes, num_decision_nodes)
        # We want to combine it with decision_probs.
        # Expand decision_probs: (batch_size, num_trees, 1, num_decision_nodes)
        decision_probs = decision_probs.unsqueeze(2)
        # Expand routing_matrix: (1, 1, num_leaf_nodes, num_decision_nodes)
        M = self.routing_matrix.unsqueeze(0).unsqueeze(0)
        # For each decision node, the probability of taking the branch specified in M:
        # if M==1: use decision_probs, if M==0: use (1 - decision_probs).
        # Compute a tensor of shape (batch_size, num_trees, num_leaf_nodes, num_decision_nodes)
        path_probs = torch.pow(decision_probs, M) * torch.pow(1 - decision_probs, 1 - M)
        # Multiply along decision nodes to get routing probability for each leaf.
        # Resulting shape: (batch_size, num_trees, num_leaf_nodes)
        leaf_probs = torch.prod(path_probs, dim=3)
        
        # Normalize leaf distributions with softmax over the class dimension.
        leaf_distributions = F.softmax(self.leaf_distributions, dim=2)  # (num_trees, num_leaf_nodes, num_classes)
        
        # Compute predictions per tree: weighted sum over leaves.
        # Using Einstein summation:
        # tree_preds: (batch_size, num_trees, num_classes)
        tree_preds = torch.einsum('btn, tnc -> btc', leaf_probs, leaf_distributions)
        
        # Average predictions over trees.
        preds = torch.mean(tree_preds, dim=1)  # (batch_size, num_classes)
        return preds

    def _get_routing_matrix(self, tree_depth):
        """
        Creates a routing matrix M of shape (num_leaf_nodes, num_decision_nodes)
        for a complete binary tree of given depth.
        
        For each leaf (represented by an index from 0 to 2**tree_depth - 1),
        we convert its index into a binary string of length tree_depth.
        Each bit in that string indicates the decision at that level:
          0 for going left (we treat that as 0) and 1 for going right.
        We then fill the routing matrix in level order.
        """
        num_leaf_nodes = 2 ** tree_depth
        num_decision_nodes = num_leaf_nodes - 1
        M = torch.zeros(num_leaf_nodes, num_decision_nodes)
        for leaf in range(num_leaf_nodes):
            # Get binary code for this leaf (length = tree_depth)
            path = [int(b) for b in format(leaf, '0{}b'.format(tree_depth))]
            # Fill M row for this leaf. In a complete binary tree in level order,
            # the decisions appear sequentially for each level.
            for level in range(tree_depth):
                # The index in level order for the decision node at this level is:
                idx = (2 ** level) - 1 + leaf >> (tree_depth - level - 1) & (2 ** level - 1)
                # For simplicity, we assume a simple ordering: the first tree_depth nodes.
                # Here we fill the first tree_depth entries with the path decisions.
                M[leaf, level] = path[level]
        return M

# Now, integrate the NeuralDecisionForest as the multi-class head in your network.
class DeepNeuralDecisionForestClassifier(nn.Module):
    def __init__(self, input_dim, num_model_classes, num_trees=5, tree_depth=3):
        """
        A deep neural network that uses a differentiable decision forest for multi-class prediction.
        It keeps your binary classification head and uses a forest for the multi-class head.
        """
        super(DeepNeuralDecisionForestClassifier, self).__init__()
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
        # Replace the fully-connected multi-class head with a Neural Decision Forest.
        self.model_forest = NeuralDecisionForest(32, num_model_classes, num_trees=num_trees, tree_depth=tree_depth)
        
    def forward(self, x):
        shared_out = self.shared(x)
        binary_out = self.binary_head(shared_out)
        model_out = self.model_forest(shared_out)
        return binary_out, model_out