# data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ETTDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: tableau numpy des valeurs de forme (n_points, n_features)
        seq_len: longueur de la séquence d'entrée
        pred_len: longueur de la séquence à prédire
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        # Ici, x aura la forme (seq_len, n_features)
        x = self.data[index : index+self.seq_len]
        # Pour y, cela dépend de ce que vous souhaitez prédire.
        # Par exemple, vous pouvez prédire toutes les variables ou une variable spécifique.
        y = self.data[index+self.seq_len : index+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_etth1(file_path, seq_len=96, pred_len=24, split_ratio=0.8):
    """
    Charge le CSV ETTh1 et retourne deux datasets (train et test).
    Ici, nous utilisons plusieurs variables, par exemple :
    ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    """
    df = pd.read_csv(file_path)
    # Extraction des variables (multivarié)
    data = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values.astype(np.float32)
    
    # Normalisation : ici, il faut normaliser chaque variable (optionnellement colonne par colonne)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    # Pour un dataset multivarié, la découpe en séquences devra tenir compte
    # de la dimension feature. Par exemple, vous pouvez découper selon l'axe 0.
    # Ici, chaque échantillon sera de forme (seq_len, n_features)
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Vous devrez adapter la classe ETTDataset pour gérer un tableau 2D par exemple,
    # ici on suppose que chaque valeur est un vecteur, donc pas besoin d'ajouter unsqueeze.
    train_dataset = ETTDataset(train_data, seq_len, pred_len)
    test_dataset = ETTDataset(test_data, seq_len, pred_len)
    
    return train_dataset, test_dataset
