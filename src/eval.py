import os
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.special
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    roc_curve,
    auc,
)
from ldc.lisa.noise import get_noise_model  

def evaluate_model(model, data_loader):
    """
    Evaluates the model on the given DataLoader and returns predictions and true labels.
    """
    model.eval()
    binary_preds = []
    binary_true = []
    overall_best_model = []
    best_submodel = []
    model_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            if model.__class__.__name__ == 'MultiTaskClassifier_L':
                if len(batch) == 4:
                    inputs, binary_labels, model_labels, _ = batch
                else:
                    inputs, binary_labels, model_labels = batch
                out_binary, out_model = model(inputs)
            elif len(batch) == 3:
                inputs, binary_labels, _ = batch
                out_binary = model(inputs)
            else:
                inputs, binary_labels = batch
                out_binary = model(inputs)
                
            
            
            # Get probabilities for the binary task.
            binary_prob = out_binary.detach().cpu().numpy().flatten()
            
            if model.__class__.__name__ == 'MultiTaskClassifier_L':
                preds_models = out_model.detach().cpu().numpy()
                model1 = preds_models[:, :5]   
                model2 = preds_models[:, 5:9]   
                model3 = preds_models[:, 9:11]
                model1_prob = scipy.special.softmax(model1, axis=1)  # Shape: (batch_size, 5), each row sums to 1.
                model2_prob = scipy.special.softmax(model2, axis=1)  # Shape: (batch_size, 4), each row sums to 1.
                model3_prob = scipy.special.softmax(model3, axis=1)  # Shape: (batch_size, 2), each row sums to 1.  
                sum_model1 = model1_prob.sum(axis=1) 
                sum_model2 = model2_prob.sum(axis=1)
                sum_model3 = model3_prob.sum(axis=1)

                group_sums = np.stack([sum_model1, sum_model2, sum_model3], axis=1)  # shape: (batch_size, 3)
        
                # Determine which model group has the highest overall probability for each sample.
                # 0 -> Model 1, 1 -> Model 2, 2 -> Model 3.
                best_group_indices = group_sums.argmax(axis=1)
                
                overall_best_model_batch = np.where(best_group_indices == 0, "Model 1",
                                    np.where(best_group_indices == 1, "Model 2", "Model 3"))
                
                best_submodel_batch = []
                for i in range(preds_models.shape[0]):
                    if best_group_indices[i] == 0:
                        sub_idx = model1[i].argmax()  
                        best_submodel_batch.append(f"1.{sub_idx}")  
                    elif best_group_indices[i] == 1:
                        sub_idx = model2[i].argmax()  
                        best_submodel_batch.append(f"2.{sub_idx+1}") 
                    else:
                        sub_idx = model3[i].argmax()  
                        best_submodel_batch.append(f"3.{sub_idx}")  
        
                best_submodel_batch = np.char.add("Model ", best_submodel_batch)
            
            binary_preds.extend(binary_prob)
            binary_true.extend(binary_labels.detach().cpu().numpy().flatten())
            if model.__class__.__name__ == 'MultiTaskClassifier_L':
                overall_best_model.extend(overall_best_model_batch)
                best_submodel.extend(best_submodel_batch)
                model_true.extend(model_labels.detach().cpu().numpy())
        if model.__class__.__name__ == 'MultiTaskClassifier_L':   
            return (np.array(binary_preds), np.array(binary_true),
                np.array(overall_best_model), np.array(best_submodel), np.array(model_true))
        return (np.array(binary_preds), np.array(binary_true))


def compute_metrics(binary_preds, binary_true, threshold=0.5, logger=None):
    """
    Computes various classification metrics for binary predictions and logs/prints them if a logger is provided.

    Metrics computed:
      - Accuracy
      - Precision
      - Recall (Sensitivity)
      - Specificity
      - F1 Score
      - ROC AUC
      - Balanced Accuracy
      - Confusion Matrix

    Args:
      binary_preds: Array of predicted probabilities.
      binary_true: Array of true binary labels.
      threshold: Threshold to convert probabilities to binary predictions.
      logger: Optional logger instance. If provided, metrics are logged and printed.

    Returns:
      A dictionary containing the computed metrics.
    """
    # Convert probabilities to binary predictions using the specified threshold.
    binary_pred_labels = (binary_preds > threshold).astype(int)

    # Compute confusion matrix.
    cm = confusion_matrix(binary_true, binary_pred_labels)

    # Compute standard metrics.
    accuracy = accuracy_score(binary_true, binary_pred_labels)
    precision = precision_score(binary_true, binary_pred_labels, zero_division=0)
    recall = recall_score(
        binary_true, binary_pred_labels, zero_division=0
    )  # Recall is sensitivity.
    f1 = f1_score(binary_true, binary_pred_labels, zero_division=0)
    roc_auc = roc_auc_score(binary_true, binary_preds)
    balanced_acc = balanced_accuracy_score(binary_true, binary_pred_labels)

    # Manually compute sensitivity and specificity from the confusion matrix.
    # Assume confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Compile metrics into a dictionary.
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
    }

    # Log and print the metrics if a logger is provided.
    if logger:
        logger.info("Validation Metrics:")
        logger.info("Accuracy: %.4f", accuracy)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall (Sensitivity): %.4f", recall)
        logger.info("Specificity: %.4f", specificity)
        logger.info("F1 Score: %.4f", f1)
        logger.info("ROC AUC: %.4f", roc_auc)
        logger.info("Balanced Accuracy: %.4f", balanced_acc)
        logger.info("Confusion Matrix:\n%s", cm)

    return metrics


def plot_confusion_matrix(cm, save_dir=None):
    """
    Plots the confusion matrix and saves it to the specified directory if provided.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["DWD", "NSWD"])
    plt.yticks(tick_marks, ["DWD", "NSWD"])
    thresh_value = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh_value else "black",
        )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
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
    Plots separate histograms of the predicted probabilities for each true class.
    For binary_true==0, the class is assumed to be DWD; for binary_true==1, NSWD.

    Args:
      binary_preds: Array of predicted probabilities.
      binary_true: Array of true binary labels.
      save_dir: Optional directory where the plots will be saved.
    """
    # Plot for DWD (binary_true == 0)
    plt.figure(figsize=(8, 6))
    plt.hist(binary_preds[binary_true == 0], bins=50, alpha=0.7, color="blue")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predicted Probabilities for DWD")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        dwd_save_path = os.path.join(save_dir, "histogram_DWD.png")
        plt.savefig(dwd_save_path)
        print(f"Histogram for DWD saved to {dwd_save_path}")
    else:
        plt.show()
    plt.close()

    # Plot for NSWD (binary_true == 1)
    plt.figure(figsize=(8, 6))
    plt.hist(binary_preds[binary_true == 1], bins=50, alpha=0.7, color="orange")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predicted Probabilities for NSWD")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        nswd_save_path = os.path.join(save_dir, "histogram_NSWD.png")
        plt.savefig(nswd_save_path)
        print(f"Histogram for NSWD saved to {nswd_save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc(binary_preds, binary_true, save_dir=None):
    """
    Plots the ROC curve and saves the figure.
    """
    fpr, tpr, thresholds = roc_curve(binary_true, binary_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance level")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

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
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    if val_loss is not None:
        plt.plot(val_loss, label="Validation Loss")
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


def plot_se_sp_vs_threshold(binary_preds, binary_true, save_dir=None):
    """
    Plots sensitivity and specificity as a function of threshold.

    Args:
      binary_preds: Array of predicted probabilities.
      binary_true: Array of true binary labels.
      save_dir: Optional directory where the plot will be saved.
    """
    thresholds = np.linspace(0, 1, 101)
    sensitivities = []
    specificities = []

    # Iterate over threshold values.
    for thresh in thresholds:
        y_pred = (binary_preds > thresh).astype(int)
        # Calculate true positives, false negatives, true negatives, and false positives.
        tp = np.sum((y_pred == 1) & (binary_true == 1))
        fn = np.sum((y_pred == 0) & (binary_true == 1))
        tn = np.sum((y_pred == 0) & (binary_true == 0))
        fp = np.sum((y_pred == 1) & (binary_true == 0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sens)
        specificities.append(spec)

    # Plot the SE and SP versus threshold.
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, label="Sensitivity")
    plt.plot(thresholds, specificities, label="Specificity")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Sensitivity and Specificity vs. Threshold")
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "se_sp_vs_threshold.png")
        plt.savefig(save_path)
        print(f"SE/SP vs Threshold plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix_multiclass(cm, class_names, save_dir=None, filename=None):
    """
    Plots the multi-class confusion matrix and saves it to the specified directory if provided.

    Args:
      cm: Confusion matrix array.
      class_names: List of class names.
      save_dir: Optional directory where the plot is saved.
      filename: Optional filename for the saved plot. If not provided and save_dir is given,
                a filename with a timestamp is generated.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Model Prediction)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Generate a unique filename if one is not provided.
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"confusion_matrix_model_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Multi-class confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def discretize_series(series):
    return series.apply(lambda x: round(x * 5) / 5)


def plot_corner(data, features, hue_col, save_path=None, palette=None):
    """
    Generates a half corner plot (pairplot) of the specified features, colored by the given hue column.

    Args:
      data (pd.DataFrame): DataFrame containing the data to plot.
      features (list): List of feature column names to include in the pairplot.
      hue_col (str): Column name in data to use for coloring.
      save_path (str, optional): Path to save the plot image.
      palette (dict, optional): Dictionary mapping discrete hue values to RGBA colors.
    """
    pairplot_fig = sns.pairplot(
        data, vars=features, hue=hue_col, diag_kind="kde", corner=True, palette=palette
    )
    if save_path:
        pairplot_fig.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_corner_predicted(data, features, save_path=None, palette=None):
    return plot_corner(
        data, features, hue_col="Predicted_Class", save_path=save_path, palette=palette
    )


def plot_corner_true(data, features, save_path=None, palette=None):
    return plot_corner(
        data, features, hue_col="True_Class", save_path=save_path, palette=palette
    )


def plot_metrics(metric_history, metric_name, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(metric_history, label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Epoch")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

