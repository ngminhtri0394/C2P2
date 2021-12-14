import torch
import torch.nn as nn


# GINConv model
class DTA_transfer(torch.nn.Module):
    def __init__(self, pencoder=None, dencoder=None, poutdim=64, doutdim=64, dropout=0.2):
        super(DTA_transfer, self).__init__()
        self.p_encoder = pencoder
        self.d_encoder = dencoder

        self.fc1_dta = nn.Linear(poutdim+doutdim, 1024)
        self.fc2_dta = nn.Linear(1024, 256)
        self.final_dta = nn.Linear(256, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, d, p):
        # print(d.shape)
        # print(p.shape)
        xd = self.d_encoder(d)
        xp = self.p_encoder(p)

        # print(xd.shape)
        # print(xp.shape)
        xc = torch.cat((xd, xp), 1)
        jointrep = xc
        # xc = xd * xp

        xc = self.dropout(self.relu(self.fc1_dta(xc)))
        xc = self.dropout(self.relu(self.fc2_dta(xc)))
        out = self.final_dta(xc)
        return out.view(-1), jointrep, xd, xp
