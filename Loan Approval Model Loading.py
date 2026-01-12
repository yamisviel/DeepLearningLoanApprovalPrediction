import torch
import pandas as pd
import torch.nn as nn
import joblib

# 1. Load Device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
print("Running on device:", device)

model = nn.Sequential(
    nn.Linear(16, 100),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(p=0.2),

    nn.Linear(50, 1)
)

# 2. Load the saved weights
model.load_state_dict(torch.load("best_model.pth",weights_only=True, map_location=device))
model.to(device)
model.eval()

# 3. Load Metadata and unseen data
meta = joblib.load('model_metadata.pkl')
new_df = pd.read_csv("data/new_loans.csv")
new_df = new_df[meta['continuous_columns'] + ['loan_intent', 'person_home_ownership', 'previous_loan_defaults_on_file']]

# 4. Preprocessing 
# Remove outlier from person_income
new_df.loc[new_df['person_income'] > meta['income_cap'], 'person_income'] = meta['income_median']

# Apply one hot encoding
new_df = pd.get_dummies(new_df)

# Reindex to ensure same columns as training data 
new_df = new_df.reindex(columns=meta['all_feature_columns'], fill_value=0)
new_df = new_df.astype('float32')

# Normalize data 
continuous_indices = [0, 1, 2, 3]
X_new = torch.tensor(new_df.values, dtype=torch.float32)
X_new[:, continuous_indices] = (X_new[:, continuous_indices] - torch.tensor(meta['X_mean'])) / torch.tensor(meta['X_std'])

# 5. predict
with torch.no_grad():
    X_new = X_new.to(device)
    logits = model(X_new)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.3).float()

# 6. Save Results
new_df['loan_probability'] = probabilities.cpu().numpy()
new_df['prediction'] = predictions.cpu().numpy()

new_df.to_csv("loan_predictions.csv", index=False)
print("Predictions saved to loan_predictions.csv")