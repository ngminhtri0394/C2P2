import torch
import esm
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import seq_cat

class ChembertaESMDataset(Dataset):
    def __init__(self, xd=None, xt=None, y=None, embed_ligand=None,ESMdict=None):
        self.ESMdict = ESMdict
        self.embedded_drug = []
        self.targets = []
        for t in tqdm(xt):
            self.targets.append(ESMdict[t])
        self.ys = []
        for i in tqdm(y):
            self.ys.append(i)
        print('Getting embedding SMILES feature')
        for smiles in tqdm(xd):
            self.embedded_drug.append(embed_ligand[smiles])

    def __getitem__(self, idx):
        xt = self.targets[idx]
        xd = self.embedded_drug[idx]
        y = self.ys[idx]
        return xd, xt, y

    def __len__(self):
        return len(self.ys)