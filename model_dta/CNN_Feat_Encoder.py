import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


# GINConv model
class CNN_Feat_Encoder(torch.nn.Module):
    def __init__(self, indim=26,outdim=128,dropout=0.2):
        super(CNN_Feat_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.embedding_xt = nn.Embedding(indim, 128)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, outdim)

    def forward(self, x):
        embedded_xt = self.embedding_xt(x)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        return xt