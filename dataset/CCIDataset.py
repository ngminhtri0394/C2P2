import os
import torch
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


class CCIDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='stitch',
                 xc1=None, xc2=None, y=None, smile_graph=None):
        self.dataset = dataset
        self.process(xc1, xc2, y, smile_graph)

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

    def process(self, xc1, xc2, y, smile_graph):
        assert (len(xc1) == len(xc2) and len(xc2) == len(y)), "The three lists must be the same length!"
        data_list_1 = []
        data_list_2 = []
        data_len = len(xc1)
        for i in tqdm(range(data_len)):
            smiles1 = xc1[i]
            smiles2 = xc2[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            try:
                c_size, features, edge_index = smile_graph[smiles1]
                c_size_2, features_2, edge_index_2 = smile_graph[smiles2]
                edge_index = torch.LongTensor(edge_index).transpose(1, 0)
                edge_index_2 = torch.LongTensor(edge_index_2).transpose(1, 0)
            except:
                continue
            # make the graph ready for PyTorch Geometrics GCN algorithms:

            GCNData_1 = DATA.Data(x=torch.Tensor(features),
                                  edge_index=edge_index,
                                  y=torch.FloatTensor([labels]))
            GCNData_1.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list

            GCNData_2 = DATA.Data(x=torch.Tensor(features_2),
                                  edge_index=edge_index_2,
                                  y=torch.FloatTensor([labels]))
            GCNData_2.__setitem__('c_size', torch.LongTensor([c_size_2]))

            data_list_1.append(GCNData_1)
            data_list_2.append(GCNData_2)

        if self.pre_filter is not None:
            data_list_1 = [data for data in data_list_1 if self.pre_filter(data)]
            data_list_2 = [data for data in data_list_2 if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_1 = [self.pre_transform(data) for data in data_list_1]
            data_list_2 = [self.pre_transform(data) for data in data_list_2]
        print('Graph construction done.')

        self.data1 = data_list_1
        self.data2 = data_list_2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]
