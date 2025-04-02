import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.

        Args:
            alpha (float): Weighting factor for the rare class.
            gamma (float): Focusing parameter to reduce the loss contribution from easy examples.
            reduction (str): 'mean', 'sum', or 'none' to reduce the loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw logits from the model of shape (batch_size, 1).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, 1), with values 0 or 1.
            
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Compute binary cross entropy loss with logits; do not reduce yet.
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Compute the probability of the target class: p_t = exp(-BCE_loss)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_saliency(model, dataset, num_samples=10):
    """
    Computes the average saliency for a number of samples from the dataset.
    
    Args:
      model: The trained model.
      dataset: A dataset (e.g., from val_loader.dataset or test set).
      num_samples: How many random samples to use for saliency computation.
      
    Returns:
      A tensor with the average saliency for each input parameter.
    """
    model.eval()
    saliency_list = []
    indices = np.random.randint(0, len(dataset), size=num_samples)
    criterion = nn.BCELoss()
    
    for i in indices:
        batch = dataset[i]
        test_input, binary_label = batch[0], batch[1]

        
        input_tensor = test_input.unsqueeze(0)
        input_tensor.requires_grad_()
        
        out_binary, _ = model(input_tensor)
        
        target = torch.tensor([binary_label], dtype=torch.float32).unsqueeze(1)
        loss = criterion(out_binary, target)
        
        loss.backward()
        
        saliency_list.append(input_tensor.grad.data.abs())
    
    saliency = torch.cat(saliency_list).cpu()
    avg_saliency = torch.mean(saliency, dim=0)
    params_impact = avg_saliency / torch.min(avg_saliency)
    params_std = torch.std(saliency, dim=0) / torch.min(avg_saliency)
    # params_impact: gives the relative impact for each parameter on the NN
    return avg_saliency, params_impact, params_std

def train_model(model, name_of_run, train_loader, val_loader=None, focal_loss_config=None, num_epochs=100, lr=0.001, logger=None):
    # Check if custom weights are provided by inspecting the first batch.
    first_batch = next(iter(train_loader))
    use_custom_weights = (len(first_batch) == 4)
    alpha = focal_loss_config.get("alpha", 0.25)
    gamma = focal_loss_config.get("gamma", 2.0)
    # Use FocalLoss for the binary classification task.
    if use_custom_weights:
        criterion_binary = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        criterion_model = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion_binary = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        criterion_model = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    if logger is None:
        logger = print
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_path = None
    
    logger("Starting training for {} epochs.".format(num_epochs))
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in epoch_bar:
            optimizer.zero_grad()
            # Unpack the batch: if custom weights are provided, batch has 4 elements.
            if use_custom_weights:
                inputs, binary_labels, model_labels, sample_weights = batch
            else:
                inputs, binary_labels, model_labels = batch
            
            out_binary, out_model = model(inputs)
            
            if use_custom_weights:
                # Compute per-sample losses.
                loss_binary = criterion_binary(out_binary, binary_labels).squeeze(1)  # shape: [batch_size]
                loss_model = criterion_model(out_model, model_labels)  # shape: [batch_size]
                combined_loss = loss_binary + loss_model
                # Multiply each sample's loss by its corresponding weight and then average.
                loss = (combined_loss * sample_weights.to(combined_loss.device)).mean()
            else:
                loss_binary = criterion_binary(out_binary, binary_labels)
                loss_model = criterion_model(out_model, model_labels)
                loss = loss_binary + loss_model
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        logger("Epoch [{}/{}] - Training Loss: {:.4f}".format(epoch+1, num_epochs, avg_train_loss))
        
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if use_custom_weights:
                        inputs, binary_labels, model_labels, _ = batch
                    else:
                        inputs, binary_labels, model_labels = batch
                    out_binary, out_model = model(inputs)
                    if use_custom_weights:
                        loss_binary = criterion_binary(out_binary, binary_labels).squeeze(1)
                        loss_model = criterion_model(out_model, model_labels)
                        loss = (loss_binary + loss_model).mean()
                    else:
                        loss_binary = criterion_binary(out_binary, binary_labels)
                        loss_model = criterion_model(out_model, model_labels)
                        loss = loss_binary + loss_model
                    running_val_loss += loss.item()
            avg_val_loss = running_val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            logger("Epoch [{}/{}] - Validation Loss: {:.4f}".format(epoch+1, num_epochs, avg_val_loss))
            
            # Save best model based on validation loss.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join("models", f"best_multi_task_classifier_{name_of_run}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger("Epoch [{}] - Best model updated (Validation Loss: {:.4f})".format(epoch+1, avg_val_loss))
    
    if val_loader is not None:
        return {
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "best_model_path": best_model_path
        }
    else:
        return {
            "train_loss_history": train_loss_history
        }