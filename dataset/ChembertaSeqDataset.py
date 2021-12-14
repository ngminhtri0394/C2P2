import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import seq_cat

class ChembertaSeqDataset(Dataset):
    def __init__(self, xd=None, xt=None, y=None, embed_ligand=None):
        self.embedded_drug = []
        self.targets = []
        for t in tqdm(xt):
            self.targets.append(torch.LongTensor([seq_cat(t.strip())]))
        self.ys = []
        for i in tqdm(y):
            self.ys.append(torch.FloatTensor([i]))
        print('Getting embedding SMILES feature')
        for smiles in tqdm(xd):
            self.embedded_drug.append(torch.from_numpy(embed_ligand[smiles]))

    def __getitem__(self, idx):
        xt = self.targets[idx]
        xd = self.embedded_drug[idx]
        y = self.ys[idx]
        return [xd, xt[0], y]

    def __len__(self):
        return len(self.ys)
