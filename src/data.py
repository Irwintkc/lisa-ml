import os
import numpy as np
import pickle
import pandas as pd
import torch
import re
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

def load_fim_data(folder_path):
    """
    Loads all pickle files from the specified folder and returns a dictionary
    with keys formatted as "Model X.Y" and values as the loaded data.

    Parameters:
      folder_path (str): Path to the folder containing the pickle files.

    Returns:
      dict: Dictionary where keys are formatted model strings and values are the loaded data.
    """
    loaded_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            file_path = os.path.join(folder_path, filename)
            # Format the model string from the filename.
            formatted = re.sub(r".*?(Model)(\d+)_(\d+).*", r"\1 \2.\3", filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            loaded_data[formatted] = data
    return loaded_data

def generate_sample_dataframe(resolved_df, loaded_fim_data, n_samples_class0, n_samples_class1):
    """
    Generates samples using the FIM for each row in resolved_df and returns a DataFrame of all samples.
    
    Parameters:
      resolved_df (pd.DataFrame): DataFrame containing binary systems with required columns.
      loaded_fim_data (dict): Dictionary of FIM matrices keyed by model and binary name.
      n_samples_class0 (int): Number of samples to generate for Class 0.
      n_samples_class1 (int): Number of samples to generate for Class 1.
      
    Returns:
      final_df (pd.DataFrame): DataFrame containing the generated samples, along with the binary Name and Model.
    """
    
    def sample_from_FIM(row, loaded_fim_data, n_samples_class0, n_samples_class1):
        # Exclude rows where the model ends with ".0"
        if row['Model'].endswith('.0'):
            return None
        
        mean = np.array([
            row['Frequency_mHz'],            
            row['FrequencyDerivative_log10'],
            row['EclipticLatitude_sin'],    
            row['EclipticLongitude'],            
            row['Amplitude_log10'],           
            row['Inclination_cos'],         
            row['Polarization'],                 
            row['InitialPhase']                  
        ])
        
        model_key = row['Model']
        try:
            fim = loaded_fim_data[model_key][row['Name']]
        except KeyError:
            raise KeyError(f"FIM not found for model '{model_key}' and source '{row['Name']}'")
        
        cov = np.linalg.inv(fim)
        
        if row['Class'] == 'DWD':
            n_samples = n_samples_class0
        elif row['Class'] == 'NSWD':
            n_samples = n_samples_class1
        else:
            n_samples = n_samples_class0
        cov = (cov + cov.T) / 2
        samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        return samples
    

    samples_dict = {}
    for idx, row in resolved_df.iterrows():
        samples = sample_from_FIM(row, loaded_fim_data, n_samples_class0, n_samples_class1)
        if samples is None:
            continue  # Skip excluded rows

        key = (row['Name'], row['Model'])
        samples_dict[key] = {'samples': samples, 'Name': row['Name'], 'Model': row['Model']}
    
    parameter_names = ['Frequency_mHz', 'FrequencyDerivative_log10', 'EclipticLatitude_sin',
                       'EclipticLongitude', 'Amplitude_log10', 'Inclination_cos', 'Polarization', 'InitialPhase']
    
    df_list = []
    for data in samples_dict.values():
        samples = data['samples']
        name = data['Name']
        model = data['Model']
        df_temp = pd.DataFrame(samples, columns=parameter_names)
        df_temp['Name'] = name
        df_temp['Model'] = model
        df_list.append(df_temp)
    
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df


def fill_with_original(group):
    '''
    Replace NaN in Sample sources with Original's Value
    '''
    group = group.sort_values(by='Source', key=lambda x: x.map({'Original': 0, 'Sample': 1}))
    original = group.iloc[0]
    return group.fillna(original)

def load_data(config):
    """
    Loads and preprocesses the data using parameters from the config.
    Splits data into training, validation, and permanent test sets,
    scales the features, saves the test set, and returns DataLoaders along with
    additional information.
    """
    
    name_of_run = config["name_of_run"]
    data_config = config["data_params"]
    resolved_folder_path = data_config["resolved_folder_path"]
    selected_features = data_config["selected_features"]
    test_seed_val = data_config["test_seed_val"]
    val_seed_val = data_config["val_seed_val"]
    training_batch_size = data_config["training_batch_size"]
    inference_batch_size = data_config["inference_batch_size"]
    ecc_0_only = data_config.get("ecc_0_only", False)
    use_custom_weights = data_config.get("use_custom_weights", False)
    n_samples_class0 = data_config["n_samples_class0"]
    n_samples_class1 = data_config["n_samples_class1"]
    fim_data_path = data_config["fim_folder_path"]
    frequency_range = data_config["frequency_range"]
    
    # Load raw data
    df = compile_data_from_folder(resolved_folder_path)
    
    # Transformations
    df['Frequency_mHz'] = df['Frequency'] * 1000
    df['FrequencyDerivative_log10'] = np.log10(df['FrequencyDerivative'])
    df['EclipticLatitude_sin'] = np.sin(df['EclipticLatitude'])
    df['Amplitude_log10'] = np.log10(df['Amplitude'])
    df['Inclination_cos'] = np.cos(df['Inclination'])
    
    # Remove unneeded original columns.
    cols_to_remove = ['Frequency', 'FrequencyDerivative', 'EclipticLatitude', 'Amplitude', 'Inclination']
    df.drop(columns=cols_to_remove, inplace=True)
    
    if frequency_range is not None:
        df = df[df['Frequency_mHz'] >= frequency_range]
        
    df['Class'] = df['Name'].str.extract(r'MW_(DWD|NSWD)')
    df = df.dropna(subset=selected_features + ['Class', 'Model'])
    
    if ecc_0_only and "Eccentricity" in selected_features:
        df = df[df["Eccentricity"] == 0]
    
    # Encode class labels.
    binary_encoder = LabelEncoder()
    df['Class_enc'] = binary_encoder.fit_transform(df['Class'])
    
    model_encoder = LabelEncoder()
    df['Model_enc'] = model_encoder.fit_transform(df['Model'])
    
    
    # Save encoders for later use.
    encoder_dir = os.path.join("models", "encoders")
    os.makedirs(encoder_dir, exist_ok=True)
    with open(os.path.join(encoder_dir, "binary_encoder.pkl"), "wb") as f:
        pickle.dump(binary_encoder, f)
    with open(os.path.join(encoder_dir, "model_encoder.pkl"), "wb") as f:
        pickle.dump(model_encoder, f)
    
    # Split data: Use 20% as a permanent test set; remaining 80% for training+validation.
    X = df[selected_features]
    y_binary = df['Class_enc']
    y_model = df['Model_enc']
    
    # For the overall split, if custom weighting is not yet applied, assign 1 as default.
    default_weights = pd.Series(np.ones(len(df)), index=df.index)
    
    X_train_val, X_test, y_binary_train_val, y_binary_test, y_model_train_val, y_model_test, _ = train_test_split(
        X, y_binary, y_model, default_weights, test_size=0.2, random_state=test_seed_val
    )
    
    # Further split training+validation into training and validation sets.
    X_train, X_val, y_binary_train, y_binary_val, y_model_train, y_model_val, weights_val = train_test_split(
        X_train_val, y_binary_train_val, y_model_train_val, default_weights.loc[X_train_val.index],
        test_size=0.2, random_state=val_seed_val
    )
    
    # ----- FIM-Based Augmentation for Training Set Only -----
    if n_samples_class0 and n_samples_class1 is not None:
        train_df = X_train.copy()
        train_df['Class_enc'] = y_binary_train.values
        train_df['Model_enc'] = y_model_train.values

        # Generate synthetic samples via FIM.
        loaded_fim_data = load_fim_data(fim_data_path)
        sample_df = generate_sample_dataframe(train_df, loaded_fim_data, n_samples_class0, n_samples_class1)
        
        train_df['Source'] = 'Original'
        sample_df['Source'] = 'Sample'
    
        train_df = pd.concat([train_df, sample_df], ignore_index=True)
        train_df = train_df.groupby('Name').apply(fill_with_original).reset_index(drop=True)
        
        if use_custom_weights:
            train_df['Weights'] = train_df.groupby('Name')['Name'].transform(lambda x: 1.0/len(x))
        
        # Re-split augmented DataFrame into features and targets, with weights if available.
        X_train = train_df[selected_features]
        y_binary_train = train_df['Class_enc']
        y_model_train = train_df['Model_enc']
        if use_custom_weights:
            weights_train = train_df['Weights']
    else:
        # If FIM augmentation is not used and custom weights are desired,
        # apply the weighting function on the original training set.
        if use_custom_weights:
            train_df = X_train.copy()
            train_df['Class_enc'] = y_binary_train.values
            train_df['Model_enc'] = y_model_train.values
            train_df['Weights'] = train_df.groupby('Name')['Name'].transform(lambda x: 1.0/len(x))
            X_train = train_df[selected_features]
            y_binary_train = train_df['Class_enc']
            y_model_train = train_df['Model_enc']
            weights_train = train_df['Weights']
    
    
    # Scale the features; fit scaler only on training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaled test set.
    test_set_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_set_df['Class'] = y_binary_test.values
    test_set_df['Model_enc'] = y_model_test.values
    test_set_df['Model'] = model_encoder.inverse_transform(y_model_test.astype(int))
    test_set_save_path = os.path.join("data", "test_set", f"test_set_scaled_seed_{test_seed_val}_{name_of_run}.csv")
    os.makedirs(os.path.dirname(test_set_save_path), exist_ok=True)
    test_set_df.to_csv(test_set_save_path, index=False)
    
    # Also, save the unscaled version.
    X_test_original = scaler.inverse_transform(X_test_scaled)
    test_set_original_df = pd.DataFrame(X_test_original, columns=X.columns)
    test_set_original_df['Class'] = y_binary_test.values
    test_set_original_df['Model_enc'] = y_model_test.values
    test_set_original_df['Model'] = model_encoder.inverse_transform(y_model_test.astype(int))
    
    test_set_original_save_path = os.path.join("data", "test_set", f"test_set_original_seed_{test_seed_val}_{name_of_run}.csv")
    os.makedirs(os.path.dirname(test_set_original_save_path), exist_ok=True)
    test_set_original_df.to_csv(test_set_original_save_path, index=False)
    
    # Convert training and validation sets into PyTorch tensors.
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_binary_train_tensor = torch.tensor(y_binary_train.values, dtype=torch.float32).unsqueeze(1)
    y_binary_val_tensor = torch.tensor(y_binary_val.values, dtype=torch.float32).unsqueeze(1)
    y_model_train_tensor = torch.tensor(y_model_train.values, dtype=torch.long)
    y_model_val_tensor = torch.tensor(y_model_val.values, dtype=torch.long)
    
    if use_custom_weights:
        weights_train_tensor = torch.tensor(weights_train.values, dtype=torch.float32)
        # For validation, use default weights (or compute custom weights separately if desired)
        weights_val_tensor = torch.tensor(np.ones(len(y_binary_val)), dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_binary_train_tensor, y_model_train_tensor, weights_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_binary_val_tensor, y_model_val_tensor, weights_val_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, y_binary_train_tensor, y_model_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_binary_val_tensor, y_model_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=inference_batch_size, shuffle=False)
    
    input_dim = X_train_tensor.shape[1]
    num_model_classes = len(model_encoder.classes_)
    
    return train_loader, val_loader, input_dim, num_model_classes, scaler, binary_encoder, model_encoder
