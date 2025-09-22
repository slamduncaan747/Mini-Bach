import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import FeedForwardNN

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.manual_seed(8)
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Hyperparameters
epochs = 1000
checkpoint_iters = 100
val_iters = 5
val_ratio = .1
batch_size = 32
learning_rate = 0.002
hidden_size = 500
dropout_rate = .1
weight_decay = 1e-5

# Create data loaders
t_set = TensorDataset(
    torch.tensor([item['input'] for item in dataset], dtype=torch.float32),
    torch.tensor([item['output'] for item in dataset], dtype=torch.float32)
)
t_size = int(len(t_set)*(1-val_ratio))
v_size = int(len(t_set)-t_size)
train_set, val_set = random_split(t_set, [t_size, v_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Define model, loss function, and optimizer
model = FeedForwardNN(input_size=1344, hidden_size=hidden_size, output_size=4480, dropout_rate=dropout_rate)
model.to(device)
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    # Train on train_set
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_function(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= batch_size

    # Save checkpoint at intervals
    if (epoch+1) % checkpoint_iters == 0:
        path = f'checkpoints/checkpoint{epoch+1}.pth'
        torch.save(model.state_dict(), path)
    
    # Get validation loss at intervals
    if (epoch+1) % val_iters == 0:
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits_val = model(x_val)
                loss_val = loss_function(logits_val, y_val)
                total_val_loss += loss_val.item()
        total_val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Val Loss: {total_val_loss:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")