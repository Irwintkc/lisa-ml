import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader=None, num_epochs=100, lr=0.001, logger=None):
    """
    Trains the model using the provided training DataLoader and optionally evaluates on a validation DataLoader.
    Saves the best model (based on validation loss) and returns training and validation loss histories.

    Args:
      model: The neural network model.
      train_loader: DataLoader for training data.
      val_loader: (Optional) DataLoader for validation data.
      num_epochs: Number of training epochs.
      lr: Learning rate for the optimizer.
      logger: Logging function (e.g. logger.info). If None, prints output.

    Returns:
      A dictionary containing:
        - train_loss_history: List of training losses per epoch.
        - val_loss_history: (If val_loader provided) List of validation losses per epoch.
        - best_model_path: (If val_loader provided) Path to the best saved model.
    """
    criterion_binary = nn.BCELoss()       # For binary output.
    criterion_model = nn.CrossEntropyLoss() # For multi-class output.
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    if logger is None:
        logger = print  # Use print if no logger is provided.
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_path = None
    
    logger("Starting training for {} epochs.".format(num_epochs))
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, binary_labels, model_labels in epoch_bar:
            optimizer.zero_grad()
            out_binary, out_model = model(inputs)
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
                for inputs, binary_labels, model_labels in val_loader:
                    out_binary, out_model = model(inputs)
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
                best_model_path = os.path.join("models", "best_multi_task_classifier.pth")
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