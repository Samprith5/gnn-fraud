import torch
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)
