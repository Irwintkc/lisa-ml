import os
import logging
import torch
import pandas as pd
import argparse
from model import MultiTaskClassifier, MultiTaskClassifier_L
from data import load_config, load_data
from train import train_model
from eval import evaluate_model, plot_confusion_matrix, plot_histogram, plot_roc, compute_metrics, plot_loss

def main(config_path):
    # Load configuration from the YAML file.
    config = load_config(config_path)
    name_of_run = config["name_of_run"]

    # Configure logging.
    log_dir = os.path.join("models")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(log_dir, "training.log"),
        filemode="w"
    )
    logger = logging.getLogger()

    # Also log to console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Starting multi-task script in main.py.")

    # Load data using configuration parameters.
    train_loader, val_loader, input_dim, num_model_classes, scaler, binary_encoder, model_encoder = load_data(config)
    logger.info("Data loaded. Input dimension: %d, Number of model classes: %d", input_dim, num_model_classes)

    # Define the model.
    model_instance = MultiTaskClassifier_L(input_dim, num_model_classes)
    logger.info("Model defined with input dimension %d and %d model classes.", input_dim, num_model_classes)

    # Check config flag to either train or load a pre-trained model.
    if config.get("test", False):
        # Skip training; load a saved model for evaluation.
        best_model_path = os.path.join("models", "best_multi_task_classifier.pth")
        if os.path.exists(best_model_path):
            model_instance.load_state_dict(torch.load(best_model_path))
            logger.info("Test flag is True. Loaded model from %s", best_model_path)
        else:
            logger.error("Test flag is True but model file %s not found.", best_model_path)
            return
    else:
        # Train the model.
        num_epochs = config["training_params"]["num_epochs"]
        lr = config["training_params"]["lr"]
        training_results = train_model(model_instance, train_loader, val_loader, num_epochs=num_epochs, lr=lr, logger=logger.info)
        
        if "best_model_path" in training_results and training_results["best_model_path"]:
            best_model_path = training_results["best_model_path"]
            model_instance.load_state_dict(torch.load(best_model_path))
            logger.info("Best model loaded from %s", best_model_path)
        if "val_loss_history" in training_results:
            loss_history_df = pd.DataFrame({
                "train_loss": training_results["train_loss_history"],
                "val_loss": training_results["val_loss_history"]
            })
        else:
            loss_history_df = pd.DataFrame({
                "train_loss": training_results["train_loss_history"]
            })

    loss_history_path = os.path.join("models", f"train_val_loss_history_{name_of_run}.csv")
    loss_history_df.to_csv(loss_history_path, index=False)
    logger.info("Training and Validation loss history saved to %s", loss_history_path)
    # Evaluate on the validation set.
    fig_dir = os.path.join("figure", name_of_run)
    os.makedirs(fig_dir, exist_ok=True)
    # Evaluate on the validation set.
    binary_preds, binary_true, model_preds, model_true = evaluate_model(model_instance, val_loader)
    metrics = compute_metrics(binary_preds, binary_true, threshold=0.5)

    # Log and print metrics.
    logger.info("Validation Metrics:")
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Sensitivity: %.4f", metrics["sensitivity"])
    logger.info("Specificity: %.4f", metrics["specificity"])
    logger.info("Confusion Matrix:\n%s", metrics["confusion_matrix"])

    print("Validation Metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("Sensitivity:", metrics["sensitivity"])
    print("Specificity:", metrics["specificity"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])

    # Plot and save evaluation graphs.
    plot_confusion_matrix(metrics["confusion_matrix"], save_dir=fig_dir)
    plot_histogram(binary_preds, binary_true, save_dir=fig_dir)
    plot_roc(binary_preds, binary_true, save_dir=fig_dir)

    loss_history_path = os.path.join("models", f"train_val_loss_history_{name_of_run}.csv")
    loss_history_df = pd.read_csv(loss_history_path)
    train_loss_history = loss_history_df['train_loss']
    val_loss_history = loss_history_df['val_loss']
    
    plot_loss(train_loss_history, val_loss_history, save_dir=fig_dir)

    logger.info("Evaluation completed successfully in main.py.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-task Training and Evaluation Script")
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Path to configuration file.')
    args = parser.parse_args()
    main(args.config)