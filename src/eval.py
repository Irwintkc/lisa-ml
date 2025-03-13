import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc


def evaluate_model(model, data_loader):
    """
    Evaluates the model on the given DataLoader and returns predictions and true labels.
    """
    model.eval()
    binary_preds = []
    binary_true = []
    model_preds = []
    model_true = []
    
    with torch.no_grad():
        for inputs, binary_labels, model_labels in data_loader:
            out_binary, out_model = model(inputs)
            # Get probabilities for the binary task.
            binary_prob = out_binary.detach().cpu().numpy().flatten()
            # For multi-class, get predicted class.
            preds_model = out_model.argmax(dim=1).detach().cpu().numpy()
            
            binary_preds.extend(binary_prob)
            binary_true.extend(binary_labels.detach().cpu().numpy().flatten())
            model_preds.extend(preds_model)
            model_true.extend(model_labels.detach().cpu().numpy())
    
    return (np.array(binary_preds), np.array(binary_true),
            np.array(model_preds), np.array(model_true))

def compute_metrics(binary_preds, binary_true, threshold=0.5):
    """
    Computes accuracy, sensitivity, specificity, and confusion matrix for binary predictions.
    """
    binary_pred_labels = (binary_preds > threshold).astype(int)
    cm = confusion_matrix(binary_true, binary_pred_labels)
    accuracy = accuracy_score(binary_true, binary_pred_labels)
    
    # cm = [[TN, FP],
    #       [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, save_dir=None):
    """
    Plots the confusion matrix and saves it to the specified directory if provided.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    thresh_value = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh_value else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_histogram(binary_preds, binary_true, save_dir=None):
    """
    Plots histograms of the predicted probabilities for each true class and saves the figure.
    """
    plt.figure(figsize=(8,6))
    plt.hist(binary_preds[binary_true == 0], bins=20, alpha=0.5, label='Negative')
    plt.hist(binary_preds[binary_true == 1], bins=20, alpha=0.5, label='Positive')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of Predicted Probabilities by True Class')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "histogram.png")
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_roc(binary_preds, binary_true, save_dir=None):
    """
    Plots the ROC curve and saves the figure.
    """
    fpr, tpr, thresholds = roc_curve(binary_true, binary_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_loss(train_loss, val_loss=None, save_dir=None):
    """
    Plots the training (and optionally validation) loss curves and saves the figure.
    """
    plt.figure(figsize=(10,6))
    plt.plot(train_loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()
    plt.close()