import torch
import torch.nn as nn

class MinMaxNorm(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(MinMaxNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        min_val = torch.min(x, dim=1, keepdim=True).values
        max_val = torch.max(x, dim=1, keepdim=True).values
        return (x - min_val) / (max_val - min_val + self.epsilon)
