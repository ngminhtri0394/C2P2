import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


# GINConv model
class BasicBlock(torch.nn.Module):
    def __init__(self, indim, hiddendim):
        super(BasicBlock, self).__init__()
        self.conv = GINConv(Sequential(Linear(indim, hiddendim), ReLU(), Linear(hiddendim, hiddendim)))
        self.bn = torch.nn.BatchNorm1d(hiddendim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        x = self.bn(x)
        return x


class GIN_Encoder(torch.nn.Module):
    def __init__(self, nlayer=5, indim=78, hiddendim=128, outdim=64, dropout=0.2):
        super(GIN_Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.indim = indim
        self.hiddendim = hiddendim

        self.nlayer = nlayer
        self.fc = Linear(hiddendim, outdim)
        self.layers = self.makelayers()

    def makelayers(self):
        layers = []
        layers.append(BasicBlock(self.indim, self.hiddendim))
        for i in range(self.nlayer-1):
            layers.append(BasicBlock(self.hiddendim, self.hiddendim))
        return nn.ModuleList(layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = layer(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x
