import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
from ultils import compile_data_from_folder

def load_config(config_path="config.yaml"):
    """
    Loads the configuration from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data(config):
    """
    Loads and preprocesses the data using parameters from the config.
    Splits data into training, validation, and permanent test sets,
    scales the features, saves the test set, and returns DataLoaders along with
    additional information.
    """
    # Read config parameters
    name_of_run = config["name_of_run"]
    data_config = config["data_params"]
    resolved_folder_path = data_config["resolved_folder_path"]
    selected_features = data_config["selected_features"]
    test_seed_val = data_config["test_seed_val"]
    val_seed_val = data_config["val_seed_val"]
    training_batch_size = data_config["training_batch_size"]
    inference_batch_size = data_config["inference_batch_size"]

    # Retrieve data into DataFrame
    df = compile_data_from_folder(resolved_folder_path)
    df['Class'] = df['Name'].str.extract(r'MW_(DWD|NSWD)')
    df = df.dropna(subset=selected_features + ['Class', 'Model'])
    
    # Encode the binary target
    binary_encoder = LabelEncoder()
    df['Class_enc'] = binary_encoder.fit_transform(df['Class'])
    
    # Encode the model target
    model_encoder = LabelEncoder()
    df['Model_enc'] = model_encoder.fit_transform(df['Model'])
    
    # Save the encoders for future use
    encoder_dir = os.path.join("models", "encoders")
    os.makedirs(encoder_dir, exist_ok=True)
    with open(os.path.join(encoder_dir, "binary_encoder.pkl"), "wb") as f:
        pickle.dump(binary_encoder, f)
    with open(os.path.join(encoder_dir, "model_encoder.pkl"), "wb") as f:
        pickle.dump(model_encoder, f)
    
    # Split data: 20% permanent test set, 80% for training+validation
    X = df[selected_features]
    y_binary = df['Class_enc']
    y_model = df['Model_enc']
    
    X_train_val, X_test, y_binary_train_val, y_binary_test, y_model_train_val, y_model_test = train_test_split(
        X, y_binary, y_model, test_size=0.2, random_state=test_seed_val
    )
    
    # Further split the 80% into train and validation (e.g., 80/20 split)
    X_train, X_val, y_binary_train, y_binary_val, y_model_train, y_model_val = train_test_split(
        X_train_val, y_binary_train_val, y_model_train_val, test_size=0.2, random_state=val_seed_val
    )
    
    # Scale the features (fit on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the permanent test set for later evaluation
    test_set_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_set_df['Class'] = y_binary_test.values
    test_set_df['Model_enc'] = y_model_test.values
    test_set_df['Model'] = model_encoder.inverse_transform(y_model_test.values)
    test_set_save_path = os.path.join("data", "test_set", f"test_set_seed_{test_seed_val}_{name_of_run}.csv")
    os.makedirs(os.path.dirname(test_set_save_path), exist_ok=True)
    test_set_df.to_csv(test_set_save_path, index=False)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_binary_train_tensor = torch.tensor(y_binary_train.values, dtype=torch.float32).unsqueeze(1)
    y_binary_val_tensor = torch.tensor(y_binary_val.values, dtype=torch.float32).unsqueeze(1)
    y_model_train_tensor = torch.tensor(y_model_train.values, dtype=torch.long)
    y_model_val_tensor = torch.tensor(y_model_val.values, dtype=torch.long)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_binary_train_tensor, y_model_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_binary_val_tensor, y_model_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=inference_batch_size, shuffle=False)
    
    input_dim = X_train_tensor.shape[1]
    num_model_classes = len(model_encoder.classes_)
    
    return train_loader, val_loader, input_dim, num_model_classes, scaler, binary_encoder, model_encoder