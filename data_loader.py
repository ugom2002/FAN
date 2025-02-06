# data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ETTDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: tableau numpy des valeurs (1D)
        seq_len: longueur de la séquence d'entrée
        pred_len: longueur de la séquence à prédire
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        x = self.data[index : index+self.seq_len]
        y = self.data[index+self.seq_len : index+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def load_etth1(file_path, seq_len=96, pred_len=24, split_ratio=0.8):
    """
    Charge le CSV ETTh1 et retourne deux datasets (train et test).
    On suppose que le CSV contient une colonne "value".
    """
    df = pd.read_csv(file_path)
    # Extraction de la série (adaptez le nom de colonne si nécessaire)
    data = df['value'].values.astype(np.float32)
    # Normalisation (moyenne et écart-type)
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    # Séparation train/test
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    train_dataset = ETTDataset(train_data, seq_len, pred_len)
    test_dataset = ETTDataset(test_data, seq_len, pred_len)
    
    return train_dataset, test_dataset
