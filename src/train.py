import os
import logging
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from ultils import compile_data_from_folder
from model import MultiTaskClassifier

# Configure logging
log_dir = os.path.join("models")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(log_dir, "training.log"),
    filemode="w"
)
logger = logging.getLogger()

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Starting multi-task training script.")

#Retrieving data into df
resolved_folder_path = '/Users/irwin/Documents/GitHub/lisa-ml/data/resolved_binaries/' #In config sometime in the future
df = compile_data_from_folder(resolved_folder_path)

df['Class'] = df['Name'].str.extract(r'MW_(DWD|NSWD)')

selected_features = ['Frequency', 'Amplitude', 'FrequencyDerivative', 'SNR', 'Eccentricity']
df = df.dropna(subset=selected_features + ['Class', 'Model'])


# Encode the binary target: DWD -> 0, NSWD -> 1
binary_encoder = LabelEncoder()
df['Class_enc'] = binary_encoder.fit_transform(df['Class'])

# Encode the model target (e.g., "Model 2.1", "Model 1.3", etc.)
model_encoder = LabelEncoder()
df['Model_enc'] = model_encoder.fit_transform(df['Model'])
logger.info("Binary Classes: %s", list(binary_encoder.classes_))
logger.info("Model Classes: %s", list(model_encoder.classes_))

encoder_dir = os.path.join("models", "encoders")
os.makedirs(encoder_dir, exist_ok=True)

# Save the binary encoder
with open(os.path.join(encoder_dir, "binary_encoder.pkl"), "wb") as f:
    pickle.dump(binary_encoder, f)

# Save the model encoder
with open(os.path.join(encoder_dir, "model_encoder.pkl"), "wb") as f:
    pickle.dump(model_encoder, f)

logger.info("Encoders saved to %s", encoder_dir)

X = df[selected_features]
y_binary = df['Class_enc']    # Binary classification target (continuous output)
y_model = df['Model_enc']       # Multi-class target for model prediction


seed_val = 1

# Train Test Split 
X_train, X_val, y_binary_train, y_binary_val, y_model_train, y_model_val = train_test_split(
    X, y_binary, y_model, test_size=0.2, random_state=seed_val
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the test set for later evaluation in a notebook
X_val_df = pd.DataFrame(X_val_scaled, columns=X.columns)
X_val_df['Class'] = y_binary_val.values
X_val_df['Model_enc'] = y_model_val.values
X_val_df['Model'] = model_encoder.inverse_transform(y_model_val.values)

test_set_save_path = os.path.join("data", "test_set", f"test_set_seed_{seed_val}.csv")
os.makedirs(os.path.dirname(test_set_save_path), exist_ok=True)
X_val_df.to_csv(test_set_save_path, index=False)
logger.info("Test set saved to %s", test_set_save_path)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_binary_train_tensor = torch.tensor(y_binary_train.values, dtype=torch.float32).unsqueeze(1)
y_binary_val_tensor = torch.tensor(y_binary_val.values, dtype=torch.float32).unsqueeze(1)
y_model_train_tensor = torch.tensor(y_model_train.values, dtype=torch.long)
y_model_val_tensor = torch.tensor(y_model_val.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_binary_train_tensor, y_model_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



input_dim = X_train_tensor.shape[1]
num_model_classes = len(model_encoder.classes_)
model_instance = MultiTaskClassifier(input_dim, num_model_classes)
logger.info("Model defined with input dimension %d and %d model classes.", input_dim, num_model_classes)


#Training config -> Plan to put it in config file
criterion_binary = nn.BCELoss()         # For binary (continuous) output
criterion_model = nn.CrossEntropyLoss()   # For multi-class model prediction
optimizer = optim.Adam(model_instance.parameters(), lr=0.001)
num_epochs = 100
train_loss_history = []

#Training Loop
logger.info("Starting training for %d epochs.", num_epochs)
for epoch in range(num_epochs):
    model_instance.train()
    running_loss = 0.0
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for inputs, binary_labels, model_labels in epoch_bar:
        optimizer.zero_grad()
        out_binary, out_model = model_instance(inputs)
        loss_binary = criterion_binary(out_binary, binary_labels)
        loss_model = criterion_model(out_model, model_labels)
        loss = loss_binary + loss_model  # You can weight these losses if needed
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Update progress bar with current loss
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_loss)
    logger.info("Epoch [%d/%d] - Combined Loss: %.4f", epoch+1, num_epochs, avg_loss)

#Output
model_save_path = os.path.join("models", "multi_task_classifier.pth")
torch.save(model_instance.state_dict(), model_save_path)
logger.info("Model saved to %s", model_save_path)

loss_history_path = os.path.join("models", "train_loss_history_multitask.csv")
pd.DataFrame(train_loss_history, columns=['Loss']).to_csv(loss_history_path, index=False)
logger.info("Training loss history saved to %s", loss_history_path)

logger.info("Training completed successfully.")