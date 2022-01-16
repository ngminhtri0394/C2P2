from model_dta.GIN_Encoder import GIN_Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# GINConv model
class CCI(torch.nn.Module):
    def __init__(self, encoder1, encoder2, n_output=2, indim=78, embed_dim=128, output_dim=128, dropout=0.2):
        super(CCI, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.fc1_cci = nn.Linear(256, 1024)
        self.fc2_cci = nn.Linear(1024, 256)
        self.out_cci = nn.Linear(256, n_output)

    def forward(self, data1, data2):
        x1 = self.encoder1(data1)
        x2 = self.encoder2(data2)

        xc = torch.cat((x1, x2), 1)
        xc = self.fc1_cci(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2_cci(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out_cci(xc)
        return out
