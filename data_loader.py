# data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ETTDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: numpy array of values with shape (n_points, n_features)
        seq_len: length of the input sequence
        pred_len: length of the prediction sequence
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        # Here, x will have the shape (seq_len, n_features)
        x = self.data[index : index+self.seq_len]
        # y can predict all variables or a specific variable depending on the task
        y = self.data[index+self.seq_len : index+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_etth1(file_path, seq_len=96, pred_len=24, split_ratio=0.8):
    """
    Load the ETTh1 CSV file and return two datasets (train and test).
    Here, we use multiple variables, for example:
    ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    """
    df = pd.read_csv(file_path)
    # Extract variables (multivariate)
    data = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values.astype(np.float32)
    
    # Normalization: normalize each variable (optionally column-wise)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    # For a multivariate dataset, sequence slicing should consider the feature dimension
    # Here, each sample will have the shape (seq_len, n_features)
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # The ETTDataset class is designed to handle a 2D array (no need to add unsqueeze)
    train_dataset = ETTDataset(train_data, seq_len, pred_len)
    test_dataset = ETTDataset(test_data, seq_len, pred_len)
    
    return train_dataset, test_dataset
