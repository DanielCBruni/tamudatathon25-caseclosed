# train_bc.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model_bc import TheMover

data = np.load("data/samples.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
y = torch.tensor(data["y"], dtype=torch.long)

try:
    data = np.load('data/samples.npz')

    print("successful")
except FileNotFoundError:
    print("Error: Make sure the 'data/samples.npz' file exists in the correct location.")
    exit()

print("--- Array Keys ---")
print(f"Keys found in the NPZ file: {list(data.keys())}")

# Access the arrays
X_loaded = data['X']
y_loaded = data['y']

# Verify the shapes
print("\n--- Shape Verification ---")
print(f"X array shape (Samples): {X_loaded.shape}")
print(f"y array shape (Labels): {y_loaded.shape}")

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = "cpu"
model = TheMover().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(30):   # takes about 10–15 minutes
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print("epoch", epoch, "loss", total)

torch.save(model.state_dict(), "data/bc_model.pth")
print("Saved model.")
