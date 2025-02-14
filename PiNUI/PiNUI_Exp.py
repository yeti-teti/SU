import os
from collections import defaultdict
import pickle

import pandas as pd
import numpy as np
import networkx as nx
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
from rich import print, progress

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score



###############################################################################
# Data Preparation
###############################################################################
def encode_sequence(seq, max_length=1000):
    # Define a mapping for common amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_int = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # reserve 0 for padding

    # Convert each amino acid in the sequence to its corresponding integer
    encoded = [aa_to_int.get(aa, 0) for aa in seq]  # default to 0 if amino acid not found

    # Pad or truncate the sequence to max_length
    if len(encoded) < max_length:
        encoded += [0] * (max_length - len(encoded))
    else:
        encoded = encoded[:max_length]
    return encoded


class PiNUIDataset(Dataset):
    def __init__(self, seqA, seqB, targets, max_length=1000):
        # Encode sequences from strings to numerical lists
        self.seqA = [encode_sequence(seq, max_length) for seq in seqA]
        self.seqB = [encode_sequence(seq, max_length) for seq in seqB]
        # Ensure targets are in a contiguous array of type float32:
        self.targets = torch.from_numpy(
            np.ascontiguousarray(np.array(targets, dtype=np.float32))
        )

        # Convert the numerical lists and targets to tensors
        self.seqA = torch.tensor(self.seqA, dtype=torch.float32)
        self.seqB = torch.tensor(self.seqB, dtype=torch.float32)
        # self.targets = torch.tensor(self.targets, dtype=torch.float32)
        # self.targets = torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {'seqA':self.seqA[idx], 'seqB':self.seqB[idx]}, self.targets[idx]
    
# Prepare dataset for training
def prepare_data(train_df, test_df, target='interaction', batch_size=32, max_length=1000):
    # Extract sequences and target values
    train_seqA = train_df['seqA'].values
    train_seqB = train_df['seqB'].values
    y_train = train_df[target].astype(np.int32).values
  
    test_seqA = test_df['seqA'].values
    test_seqB = test_df['seqB'].values
    y_test = test_df[target].astype(np.int32).values
  
    # Create datasets with encoding
    train_dataset = PiNUIDataset(train_seqA, train_seqB, y_train, max_length)
    test_dataset = PiNUIDataset(test_seqA, test_seqB, y_test, max_length)
  
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
    return train_loader, test_loader



###############################################################################
# Model Definition
###############################################################################
## MLP  
class PiNUIMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.Dropout = nn.Dropout(dropout)

    def forward(self, features_dict):
        # Embedding for each seq
        seqA = features_dict['seqA']
        seqB = features_dict['seqB']

        x = torch.cat([
            seqA,
            seqB
        ], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        
        return x
    

# Train and Evaluatie
def train_model(model, train_loader,test_loader, criterion, optimizer, num_epochs, device='cuda', early_stopping=5):

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []

    for epoch in range(num_epochs):

        # Training phase
        model.train()
        train_loss = 0.0

        for batch_features, batch_targets in progress.track(train_loader, description=f"Epoch {epoch + 1}"):
            
            batch_features = {k:v.to(device) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward() 
            optimizer.step()

            train_loss += loss.item()
    
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = {k:v.to(device) for k, v in batch_features.items()}
                batch_targets = batch_targets.to(device).unsqueeze(-1)

                outputs = model(batch_features)
                v_loss = criterion(outputs, batch_targets)
                                
                val_loss += v_loss.item()

        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)


        # Append each epoch history
        training_history.append({
            'epoch': epoch + 1, 
            'train_loss': train_loss, 
            'val_loss': val_loss
        })

        print(f'\n Epoch: {epoch + 1}/{num_epochs} ') 
        print(f'\n Training loss: {train_loss:.4f}')
        print(f'\n Validation loss: {val_loss:.4f}')

        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= early_stopping:
                print(f'Early stopping after {epoch + 1} epochs')
                break
    
    return pd.DataFrame(training_history)

def evaluate_model(model, test_loader, device='cuda'):

    model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = {k:v.to(device, non_blocking=True) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device, non_blocking=True).unsqueeze(-1)

            outputs = model(batch_features)

            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_targets.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    return predictions, actuals 


###############################################################################
# Main
###############################################################################
def main():
    print("Loading Data...")
    data_human = pd.read_csv("https://shiru-public.s3.us-west-2.amazonaws.com/PiNUI/PiNUI-human.csv")

    # Optional: check for NaN or invalid values in your dataset
    # data_human.dropna(subset=['seqA', 'seqB', 'interaction'], inplace=True)

    print("Splitting Data...")
    train_df, test_df = train_test_split(data_human, train_size=0.8, shuffle=True)

    print("Preparing Dataloaders...")
    train_loader, test_loader = prepare_data(
        train_df,
        test_df,
        target='interaction',
        batch_size=32,
        max_length=1000
    )

    # Initialize model
    print("Intializing the Model...")
    model = PiNUIMLP(
        input_dim=2000, 
        output_dim=1, 
        hidden_dim=256, 
        dropout=0.1,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Setting the device...")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        if torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")


    print("Training the model...")
    history = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=1, 
        device=device, 
        early_stopping=5
    )

    # Evaluate
    print("Evaluating the model...")
    model.load_state_dict(torch.load("best_model.pth"))
    raw_logits, true_targets = evaluate_model(model, test_loader, device=device)

    # Convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(raw_logits)).numpy().flatten()
    predicted_labels = (probs > 0.5).astype(np.int32)
    # Flatten true_targets and convert to int (if needed)
    true_labels = true_targets.astype(np.int32).flatten()

    # Debug prints
    print("True labels shape:", true_labels.shape)
    print("Probabilities shape:", probs.shape)
    print("Unique true labels:", np.unique(true_labels))
    print("Unique predicted labels:", np.unique(predicted_labels))

    # Compute classification metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, probs)  # Removed multi_class parameter
    f1 = f1_score(true_labels, predicted_labels)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")


    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        'history': history,
        'raw_logits': raw_logits,
        'true_targets': true_targets,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1': f1
    }
    with open("results/results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Results saved to 'results/results.pkl'")


if __name__ == "__main__":
    main()
