import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


# GINConv model
class ESM_Feat_Encoder(torch.nn.Module):
    def __init__(self, indim=768,outdim=128,dropout=0.2):
        super(ESM_Feat_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.projection = Linear(indim, outdim)

    def forward(self, x):
        x = self.dropout(self.relu(self.projection(x)))
        return x