from collections import defaultdict

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Dataset Loader
class PiNUIDataset(Dataset):
    def __init__(self, features, targets):
        self.seqA = torch.FloatTensor(features['seqA'])
        self.seqB = torch.FloatTensor(features['seqB'])
        self.targets = torch.FloatTensor(features['targets'])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {self.seqA[idx], self.seqB[idx]}, self.targets[idx] 
    
# Prepare dataset for training
def prepare_data(train_encoded, test_encoded, target='interaction', batch_size=32):

    protein_cols = [col for col in train_encoded.columns if col not in ['interaction']]

    X_train = train_encoded[protein_cols].values
    y_train = train_encoded[target].values
    X_test = test_encoded[protein_cols].values
    y_test = test_encoded[target].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = PiNUIDataset(X_train_scaled, y_train)
    test_dataset = PiNUIDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

    return train_loader, test_loader


class ProteinMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super.__init__()
        pass

    def forward(self, x):
        pass

def train():
    pass 

def evaluate_model():
    pass


def main():
    
    # Load dataset
    print("Loading Data...")
    data_human = pd.read_csv("https://shiru-public.s3.us-west-2.amazonaws.com/PiNUI/PiNUI-human.csv")

    train_val_proteins, test_proteins = train_test_split(data_human, train_size=0.8)
    
    # Prepare dataset
    print("Preparing dataset")
    train_loader, test_loader = prepare_data(
        train_val_proteins, test_proteins, target='interaction', batch_size=32
    )

    # Initialize model
    # Train Model
    # Evaluate the model
    # Plot results
    
    


if __name__ == "__main__":
    main()