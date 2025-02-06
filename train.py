# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import load_etth1
from models import TransformerForecast

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            # x: (batch, seq_len, input_size)
            # y: (batch, pred_len, input_size) for multivariate cases
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)  # (batch, pred_len, input_size)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return model

def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    # Parameters
    seq_len = 96          # Length of the input sequence
    pred_len = 24         # Length of the prediction window
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    d_model = 64
    nhead = 4
    num_layers = 2
    d_ff = 128
    input_size = 7        # Multivariate: for example, 7 variables (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
    # The model predicts a sequence of length pred_len for each variable, resulting in an output of shape (pred_len, input_size).

    # Load datasets (ensure 'data/ETTh1.csv' exists and is prepared for multivariate input)
    train_dataset, test_dataset = load_etth1('data/ETTh1.csv', seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    # Training and evaluation of the classic Transformer
    print("=== Training the Classic Transformer ===")
    model_classic = TransformerForecast(input_size, d_model, nhead, num_layers, d_ff, pred_len, use_fan=False)
    optimizer_classic = optim.Adam(model_classic.parameters(), lr=learning_rate)
    train_model(model_classic, train_loader, criterion, optimizer_classic, num_epochs, device)
    evaluate_model(model_classic, test_loader, criterion, device)

    # Training and evaluation of the Transformer with FAN
    print("=== Training the Transformer with FAN ===")
    model_fan = TransformerForecast(input_size, d_model, nhead, num_layers, d_ff, pred_len, use_fan=True, fan_p_ratio=0.25)
    optimizer_fan = optim.Adam(model_fan.parameters(), lr=learning_rate)
    train_model(model_fan, train_loader, criterion, optimizer_fan, num_epochs, device)
    evaluate_model(model_fan, test_loader, criterion, device)

if __name__ == '__main__':
    main()
