# model_bc.py
import torch
import torch.nn as nn
import numpy as np
from case_closed_game import Direction

class TheMover(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*18*20, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 actions
        )

    def forward(self, x):
        return self.net(x)

def preprocess(board, me_trail, enemy_trail):
    """Convert state from Flask agent into model input."""
    inp = np.zeros((2, 18, 20), dtype=np.float32)
    for x, y in me_trail:
        inp[0, y, x] = 1
    for x, y in enemy_trail:
        inp[1, y, x] = 1
    return torch.tensor(inp).unsqueeze(0)

def action_from_logits(logits):
    idx = int(logits.argmax().item())
    return list(Direction)[idx]
