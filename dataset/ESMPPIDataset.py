import numpy as np
import torch
from torch.utils.data import Dataset

class ESMPPIDataset(Dataset):
    def __init__(self,x1,x2,y,esmdict):
        self.p1seq = []
        self.p2seq = []
        print(type(esmdict))
        for x in x1:
            self.p1seq.append(esmdict[x])
        for x in x2:
            self.p2seq.append(esmdict[x])

        self.y = torch.tensor(y,dtype=torch.float32)

    def __getitem__(self, item):
        return np.array(self.p1seq[item]), np.array(self.p2seq[item]), self.y[item]

    def __len__(self):
        return len(self.y)