from tqdm import tqdm
from torch.utils.data import Dataset

class CCIChembertaDataset(Dataset):
    def __init__(self, xd1=None, xd2=None, y=None, embed_ligand=None):
        self.embedded_drug1 = []
        self.embedded_drug2 = []
        self.targets = []
        self.ys = []
        for i in tqdm(y):
            self.ys.append(i)
        print('Getting embedding SMILES feature')
        for smiles in tqdm(xd1):
            self.embedded_drug1.append(embed_ligand[smiles])
        print('Getting embedding SMILES feature')
        for smiles in tqdm(xd2):
            self.embedded_drug2.append(embed_ligand[smiles])

    def __getitem__(self, idx):
        xd1 = self.embedded_drug1[idx]
        xd2 = self.embedded_drug2[idx]
        y = self.ys[idx]
        return xd1, xd2, y

    def __len__(self):
        return len(self.ys)