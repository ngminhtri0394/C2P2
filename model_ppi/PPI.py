import torch
import torch.nn as nn
from torch.nn import Linear

class PPI(nn.Module):
    def __init__(self, encoder, dimin=768, dropout=0.1):
        super(PPI, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nonlinear = nn.ReLU()
        self.encoder = encoder
        self.fc1 = Linear(128 * 2, 128)
        self.fc2 = Linear(128, 1)

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x1)
        v_out = torch.cat((x1, x2), 1)
        v_out = self.dropout(v_out)
        v_out = self.fc1(v_out)
        v_out = self.nonlinear(v_out)
        v_out = self.dropout(v_out)
        out = self.fc2(v_out)
        return out.view(-1)