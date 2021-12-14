import torch
import torch.nn as nn
from torch.nn import Linear


# GINConv model
class Chemberta_Encoder(torch.nn.Module):
    def __init__(self, indim=768, outdim=64, dropout=0.2):
        super(Chemberta_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.project = Linear(indim, outdim)
        # self.project2 = Linear(128, outdim)

    def forward(self, x):
        x = self.dropout(self.relu(self.project(x)))
        # x = self.dropout(self.relu(self.project2(x)))
        return x
