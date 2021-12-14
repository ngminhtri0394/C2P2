import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from copy import deepcopy
from tqdm import tqdm
from tqdm import trange
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import time
import re
import esm
import networkx as nx
import pandas as pd
from rdkit import Chem
from  rdkit.Chem.rdmolfiles import MolFromMol2File
from config import *
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)

        c_size = mol.GetNumAtoms()
        c_size = mol.GetNumAtoms()
    except:
        print(smile)
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def smile_to_graph_pdbbind(smile, pdb):
    mol = Chem.MolFromSmiles(smile)
    if mol == None:
        if os.path.isfile('../../windows/GEFA/data/PDBBind/v2019-other-PL/'+pdb+'/'+pdb+'_ligand.mol2'):
            mol = MolFromMol2File('../../windows/GEFA/data/PDBBind/v2019-other-PL/'+pdb+'/'+pdb+'_ligand.mol2')
        else:
            mol = MolFromMol2File('../../windows/GEFA/data/PDBBind/refined-set/' + pdb + '/' + pdb + '_ligand.mol2')
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

class TestbedDatasetESMFeature(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, ESMdict=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDatasetESMFeature, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()
        self.ESMdict = ESMdict
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in trange(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            try:
                edge_index = torch.LongTensor(edge_index).transpose(1, 0)
            except:
                edge_index = torch.LongTensor(edge_index)
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=edge_index,
                                y=torch.FloatTensor([labels]))
            # GCNData.target = torch.LongTensor([target])
            GCNData.target = self.ESMdict[target]
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])



seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
pdbs_seq = []
for dt_name in ['PDBBind']:
    opts = ['train', 'test', 'valid']
    for opt in opts:
        df = pd.read_csv('data/' + args.dataset + '/split/' + args.dataset + '_' + opt + '_setting_' + '1' + '.csv')
        pdbs_seq += list(df['target_sequence'])
pdbs_seq = set(pdbs_seq)

esm_bert, esm_alphabet = esm.pretrained.esm1_t12_85M_UR50S()
esm_bert.to('cuda:0')
batch_converter = esm_alphabet.get_batch_converter()
num_esm_layers = len(esm_bert.layers)
seq_emb = {}
print('Preprocessing seq: ')
for seq in tqdm(pdbs_seq):
    _, _, batch_tokens = batch_converter([("", seq)])
    esmemb = esm_bert(batch_tokens.to('cuda:0'), repr_layers=[num_esm_layers])
    emb = esmemb["representations"][num_esm_layers]
    emb = emb[:, 1: len(seq)].mean(1)
    emb = emb.cpu().detach().numpy()
    seq_emb[seq] = torch.FloatTensor(emb)



for i in range(2,5):
    opts = ['train', 'test', 'valid']
    compound_iso_smiles = []
    pdbs = []
    for opt in opts:
        df = pd.read_csv('data/'+args.dataset+'/visualize_split/' + args.dataset + '_' + opt + '_setting_'+str(i)+'.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
        pdbs += list(df['target_name'])
    smilespdb = zip(list(compound_iso_smiles), list(pdbs))
    compound_iso_smiles = set(compound_iso_smiles)

    smile_graph = {}
    for smile, pdb in smilespdb:
        if args.dataset == 'PDBBind':
            smile_graph[smile] = smile_to_graph_pdbbind(smile, pdb)
        else:
            smile_graph[smile] = smile_to_graph(smile)
    # convert to PyTorch data format
    for opt in opts:
        df = pd.read_csv('data/'+args.dataset+'/visualize_split/' + args.dataset + '_'+opt+'_setting_'+str(i)+'.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [t for t in train_prots]
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), XT, np.asarray(train_Y)

        print('preparing ', args.dataset + '_'+opt+'_setting_' + str(i) + '.pt in pytorch format!')
        train_data = TestbedDatasetESMFeature(root='data/' + args.dataset,
                                              dataset=args.dataset + '_ESM_'+opt+'_setting_' + str(i) + '', xd=train_drugs,
                                              xt=train_prots, y=train_Y,
                                              smile_graph=smile_graph, ESMdict=seq_emb)




