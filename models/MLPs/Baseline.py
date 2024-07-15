import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size = 138, hidden_size = 64, output_size = 1):
        super(Model, self).__init__()
        self.verbose = 0
        self.fc1 = nn.Linear(138, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x,y,_x,_y):
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out