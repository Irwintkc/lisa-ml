import os
import logging
import torch
import numpy as np
import pandas as pd
import argparse
import pprint
import matplotlib.pyplot as plt

from model import (
    MultiTaskClassifier,
    MultiTaskClassifier_L,
    DeepNeuralDecisionForestClassifier
)
from data import load_config, load_data
from train import train_model, compute_saliency
from eval import (
    evaluate_model,
    plot_confusion_matrix,
    plot_histogram,
    plot_roc,
    compute_metrics,
    plot_loss,
    plot_se_sp_vs_threshold,
    plot_confusion_matrix_multiclass,
    plot_corner,
    discretize_series,
    plot_metrics
)
from sklearn.metrics import confusion_matrix



def main(config_path):
    # Load configuration.
    config = load_config(config_path)
    name_of_run = config["name_of_run"]
    focal_loss_config = config["training_params"].get("focal_loss", {})
    evalutaion_threshold = config["evaluation_threshold"]
    fig_dir = os.path.join("figures", name_of_run)
    
    os.makedirs(fig_dir, exist_ok=True)
    # Configure logging.
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)

    if config.get("test", False):
        log_filename = os.path.join(log_dir, f"val_{name_of_run}.log")
    else:
        log_filename = os.path.join(log_dir, f"training_{name_of_run}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="w"
    )
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Starting multi-task script in main.py.")
    logger.info("Loaded configuration settings:\n%s", pprint.pformat(config))

    # Load data.
    train_loader, val_loader, input_dim, num_model_classes, scaler, binary_encoder, model_encoder = load_data(config)
    logger.info("Data loaded. Input dimension: %d, Number of model classes: %d", input_dim, num_model_classes)

    model_mapping = {
        "NeuralForest": DeepNeuralDecisionForestClassifier,
        "Multi": MultiTaskClassifier,
        "Multi_L": MultiTaskClassifier_L
    }

    model_choice = config["model"]

    model_class = model_mapping.get(model_choice)
    if model_class is None:
        raise ValueError(f"Unknown model type specified in config: {model_choice}")

    model_instance = model_class(input_dim, num_model_classes)
    logger.info("Model defined: %s with input dimension %d and %d model classes.",
                model_instance.__class__.__name__, input_dim, num_model_classes)

    #Training/Loading Phase
    if config.get("test", False):
        best_model_path = os.path.join("models", f"best_multi_task_classifier_{name_of_run}.pth")
        if os.path.exists(best_model_path):
            model_instance.load_state_dict(torch.load(best_model_path))
            logger.info("Test flag is True. Loaded model from %s", best_model_path)
        else:
            logger.error("Test flag is True but model file %s not found.", best_model_path)
            return
    else:
        num_epochs = config["training_params"]["num_epochs"]
        lr = config["training_params"]["lr"]
        training_results = train_model(
            model_instance, name_of_run, train_loader, val_loader,
            focal_loss_config=focal_loss_config, num_epochs=num_epochs, lr=lr,
            logger=logger.info
        )
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

        loss_history_path = os.path.join("models", "loss_history", f"train_val_loss_history_{name_of_run}.csv")
        os.makedirs(os.path.dirname(loss_history_path), exist_ok=True)
        loss_history_df.to_csv(loss_history_path, index=False)
        logger.info("Training and Validation loss history saved to %s", loss_history_path)
        se_history_path = os.path.join(fig_dir, "se_history.png")
        plot_metrics(training_results["sensitivity_history"], "Sensitivity", se_history_path)
        sp_history_path = os.path.join(fig_dir, "sp_history.png")
        plot_metrics(training_results["specificity_history"], "Specificity", sp_history_path)
        acc_history_path = os.path.join(fig_dir, "acc_history.png")
        plot_metrics(training_results["accuracy_history"], "Accuracy", acc_history_path)
        mcc_history_path = os.path.join(fig_dir, "mcc_history.png")
        plot_metrics(training_results["mcc_history"], "MCC", mcc_history_path)
        auc_history_path = os.path.join(fig_dir, "auc_history.png")
        plot_metrics(training_results["auc_history"], "AUC ROC", auc_history_path)
    # Compute saliency.
    avg_saliency, params_impact, params_std = compute_saliency(model_instance, val_loader.dataset, num_samples=10)
    selected_features = config["data_params"]["selected_features"]
    avg_saliency_list = avg_saliency.tolist()
    params_impact_list = params_impact.tolist()
    params_std_list = params_std.tolist()
    for feature, sal, pimp, pstd in zip(selected_features, avg_saliency_list, params_impact_list, params_std_list):
        logger.info("Feature: %s, Average Saliency: %.4f, Parameters Impact: %.4f, Parameters Std: %.4f",
                    feature, sal, pimp, pstd)

    # Evaluate on the validation set.
    binary_preds, binary_true, overall_best_model, best_submodel, model_true = evaluate_model(model_instance, val_loader)
    model_true_labels = model_encoder.inverse_transform(model_true)
    model_true_overall_label = np.array(["Model " + label.split()[1].split('.')[0] for label in model_true_labels])
    metrics = compute_metrics(binary_preds, binary_true, threshold=evalutaion_threshold, logger=logger)

    plot_confusion_matrix(metrics["confusion_matrix"], save_dir=fig_dir)
    plot_histogram(binary_preds, binary_true, save_dir=fig_dir)
    plot_roc(binary_preds, binary_true, save_dir=fig_dir)
    plot_se_sp_vs_threshold(binary_preds, binary_true, save_dir=fig_dir)

    overall_class_names = ["Model 1", "Model 2", "Model 3"]
    cm_overall = confusion_matrix(model_true_overall_label, overall_best_model, labels=overall_class_names)
    plot_confusion_matrix_multiclass(cm_overall, class_names=overall_class_names, save_dir=fig_dir, filename="cm_model_overall.png")

    mask = (model_true_overall_label == overall_best_model)
    true_submodels = model_true_labels[mask]
    pred_submodels = np.array(best_submodel)[mask]
    submodel_classes = np.unique(np.concatenate((true_submodels, pred_submodels)))
    cm_sub = confusion_matrix(true_submodels, pred_submodels, labels=submodel_classes)
    plot_confusion_matrix_multiclass(cm_sub, class_names=list(submodel_classes), save_dir=fig_dir, filename="cm_correct_overall_submodel.png")

    try:
        loss_history_path = os.path.join("models", "loss_history", f"train_val_loss_history_{name_of_run}.csv")
        loss_history_df = pd.read_csv(loss_history_path)
        train_loss_history = loss_history_df["train_loss"]
        val_loss_history = loss_history_df["val_loss"]
        plot_loss(train_loss_history, val_loss_history, save_dir=fig_dir)
    except Exception as e:
        logger.error("Could not plot loss curve: %s", e)

    # ---- Create corner plots of the validation parameters ----
    X_val = val_loader.dataset.tensors[0].numpy()
    X_val_original = scaler.inverse_transform(X_val)
    X_val_df = pd.DataFrame(X_val_original, columns=selected_features)

    X_val_df["Predicted_Class"] = binary_preds
    X_val_df["True_Class"] = binary_true
    X_val_df["Predicted_Class_binary"] = (X_val_df["Predicted_Class"] > evalutaion_threshold).astype(int)
    X_val_df["Predicted_Class"] = discretize_series(X_val_df["Predicted_Class"])
    X_val_df["True_Class"] = discretize_series(X_val_df["True_Class"])
    X_val_df["Prediction_Status"] = np.where(
    X_val_df["Predicted_Class_binary"] == X_val_df["True_Class"],
    "Correct",
    "Wrong"
    )
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    alpha_values = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]

    cmap = plt.get_cmap("plasma", len(bins))
    custom_palette = {}
    for i, bin_val in enumerate(bins):
        r, g, b, _ = cmap(i)
        custom_palette[bin_val] = (r, g, b, alpha_values[i])

    pred_plot_path = os.path.join(fig_dir, "corner_plot_predicted.png")
    plot_corner(X_val_df, selected_features, hue_col="Predicted_Class", save_path=pred_plot_path, palette=custom_palette)
    logger.info("Corner plot for predicted classes saved to %s", pred_plot_path)

    true_plot_path = os.path.join(fig_dir, "corner_plot_true.png")
    plot_corner(X_val_df, selected_features, hue_col="True_Class", save_path=true_plot_path, palette=custom_palette)
    logger.info("Corner plot for true classes saved to %s", true_plot_path)

    prediction_status_palette = {"Correct": "green", "Wrong": "red"}
    df_class1 = X_val_df[X_val_df["True_Class"] == 1]
    
    class1_status_plot_path = os.path.join(fig_dir, "corner_plot_class1_status.png")
    plot_corner(df_class1, selected_features, hue_col="Prediction_Status", 
                save_path=class1_status_plot_path, palette=prediction_status_palette)
    logger.info("Corner plot for true class 1 (Prediction_Status) saved to %s", class1_status_plot_path)
    
    df_class0 = X_val_df[X_val_df["True_Class"] == 0]
    class0_status_plot_path = os.path.join(fig_dir, "corner_plot_class0_status.png")
    plot_corner(df_class0, selected_features, hue_col="Prediction_Status", 
                save_path=class0_status_plot_path, palette=prediction_status_palette)
    
    logger.info("Evaluation completed successfully in main.py.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-task Training and Evaluation Script")
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Path to configuration file.')
    args = parser.parse_args()
    main(args.config)