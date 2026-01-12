import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib
from torch.utils.data import random_split, DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
print("Running on device:", device)

df = pd.read_csv("data/loan_data.csv")

# --- Preprocessing ---

# Features to use 
features = ['person_income', 'loan_amnt','loan_int_rate', 'loan_percent_income',
            'loan_intent', 'person_home_ownership','previous_loan_defaults_on_file', 
            'loan_status'] 

df = df[features].copy()

# Calculating the median and 99th percentile
income_median = df['person_income'].median()
income_cap = df['person_income'].quantile(0.99)

# Apply the replacement
df.loc[df['person_income'] > income_cap, 'person_income'] = income_median

# One hot encoding for catergorial variables
df = pd.get_dummies(df, columns=["loan_intent",
                                 "person_home_ownership",
                                 "previous_loan_defaults_on_file"]).astype("float32")

# --- Feature Selection ---

target_col = 'loan_status'

X_data = df.drop(columns=[target_col]).values 
y_data = df[target_col].values

# Create tensor 
X = torch.tensor(X_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)

dataset = TensorDataset(X, y)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# normalize X as the ranges are different
continuous_indices = [0, 1, 2, 3]
train_indices = train_dataset.indices
X_train_cont = X[train_indices][:,continuous_indices]

X_mean = X_train_cont.mean(axis=0)
X_std = X_train_cont.std(axis=0) + 1e-7 # To prevent division by zero
X[:, continuous_indices] = (X[:, continuous_indices] - X_mean) / X_std

# --- Metadata ---
metadata = {
    'continuous_columns': ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'],
    'all_feature_columns': df.drop(columns=[target_col]).columns.tolist(),
    'X_mean': X_mean.cpu().numpy(),
    'X_std': X_std.cpu().numpy(),
    'income_cap': income_cap,
    'income_median': income_median
}

# Save metadata
joblib.dump(metadata, 'model_metadata.pkl')
print("Metadata saved successfully!")

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
WEIGHT_DECAY = 1e-5

# --- Early Stopping Variables ---
patience = 5          # How many epochs to wait before stopping
min_delta = 0.001     # Minimum improvement required to count as "better"
best_val_loss = float('inf') 
counter = 0           # Tracks how many epochs passed without improvement

# --- DataLoaders ---
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Model & Optimizer --- 
model = nn.Sequential(
    nn.Linear(16, 100),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(50, 1)
)
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    print(f"--- EPOCH: {epoch} ---")
    model.train()
    
    loss_sum = 0
    train_accurate = 0
    train_sum = 0
    for X_batch, y_batch in tqdm(train_dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).type(torch.float).reshape(-1, 1)

        outputs = model(X_batch)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y_batch)
        loss_sum+=loss.item()
        loss.backward()
        optimizer.step()

        predictions = torch.sigmoid(outputs) > 0.5
        accurate = (predictions == y_batch).sum().item()
        train_accurate+=accurate
        train_sum+=y_batch.size(0)

    avg_train_loss = loss_sum / len(train_dataloader)    
    train_losses.append(loss_sum / len(train_dataloader)) 
    print("Training loss: ", loss_sum / len(train_dataloader))
    print("Training accuracy: ", train_accurate / train_sum)

    model.eval()
    val_loss_sum = 0
    val_accurate = 0
    val_sum = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_dataloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).type(torch.float).reshape(-1, 1)

            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            val_loss_sum+=loss.item()

            predictions = torch.sigmoid(outputs) > 0.5
            accurate = (predictions == y_batch).sum().item()
            val_accurate+=accurate
            val_sum+=y_batch.size(0)

    avg_val_loss = val_loss_sum / len(val_dataloader)
    val_losses.append(val_loss_sum / len(val_dataloader))
    print("Validation loss: ", val_loss_sum / len(val_dataloader))
    print("Validation accuracy: ", val_accurate / val_sum)

    # Early Stopping Logic
    if avg_val_loss < (best_val_loss - min_delta):
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth") # Save the winner
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(torch.load("best_model.pth"))

plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('final.png')

model.eval()
all_preds = []
all_true = []

# Thresholds to test
thresholds = [0.5, 0.4, 0.3]

with open("classification_reports.txt", "w") as f:
    for threshold in thresholds:
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_dataloader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = (torch.sigmoid(outputs) > threshold).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_true.extend(y_batch.numpy().flatten())

        report = classification_report(all_true, all_preds)
        header = f"\n{'='*30}\nClassification Report: Threshold {threshold}\n{'='*30}\n"
        
        # Write to file
        f.write(header)
        f.write(report)
        
        print(header)
        print(report)

print("All reports have been saved to classification_reports.txt")