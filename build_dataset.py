from dataset.GraphESMFeatureDataset import GraphESMFeatureDataset
from dataset.GraphSeqDataset import GraphSeqDataset
from dataset.ChembertaSeqDataset import ChembertaSeqDataset
from dataset.ChembertaESMDataset import ChembertaESMDataset
from dataset.CCIDataset import CCIDataset
from dataset.ESMPPIDataset import ESMPPIDataset
from dataset.CCIChembertaDataset import CCIChembertaDataset
import pickle
import pandas as pd
from torch_geometric.data import DataLoader as GMDataLoader
from torch.utils.data import DataLoader as PTDataLoader
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import esm
from tqdm import tqdm
import torch
from utils import collate

class LigandTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./data/PubChem10M_SMILES_BPE_450k/")

    def tokenize(self, smiles, only_tokens=False):
        tokens = self.tokenizer(smiles)
        if not (only_tokens):
            return tokens
        return list(np.array(tokens['input_ids'], dtype="int"))

def build_dta_dataset(args):
    print('Batch size: ',args.batch_size)
    print('LR: ', args.lr)
    if 'GIN' in args.denc:
        if 'CNN' in args.penc:
            train_data = GraphSeqDataset(root='../GraphDTA/data/' + args.dataset,
                                         dataset=args.dataset + '_train_setting_' + args.setting)
            valid_data = GraphSeqDataset(root='../GraphDTA/data/' + args.dataset,
                                         dataset=args.dataset + '_valid_setting_' + args.setting)
            test_data = GraphSeqDataset(root='../GraphDTA/data/' + args.dataset,
                                        dataset=args.dataset + '_test_setting_' + args.setting)
        elif 'ESM' in args.penc:
            if args.ppipretrain:
                if args.pretrainprojection:
                    featpath = args.penc
                else:
                    featpath = args.penc + '_PPI'
            else:
                featpath = args.penc

            train_data = GraphESMFeatureDataset(root='../GraphDTA/data/' + args.dataset,
                                                dataset=args.dataset + '_train_' + featpath + '_Feat_setting_' + args.setting)
            valid_data = GraphESMFeatureDataset(root='../GraphDTA/data/' + args.dataset,
                                                dataset=args.dataset + '_valid_' + featpath + '_Feat_setting_' + args.setting)
            test_data = GraphESMFeatureDataset(root='../GraphDTA/data/' + args.dataset,
                                               dataset=args.dataset + '_test_' + featpath + '_Feat_setting_' + args.setting)
    elif 'Chemberta' in args.denc:
        with open("../GraphDTA/data/" + args.dataset + "/chembert_feature_" + args.dataset + ".pkl", 'rb') as handle:
            embedded_drug = pickle.load(handle)
        dftrain = pd.read_csv(
            '../GraphDTA/data/' + args.dataset + '/split/' + args.dataset + '_train_setting_' + str(args.setting) + '.csv')
        dftrain_drug = list(dftrain['compound_iso_smiles'])
        dftrain_target = list(dftrain['target_sequence'])
        dftrain_y = list(dftrain['affinity'])

        dfvalid = pd.read_csv(
            '../GraphDTA/data/' + args.dataset + '/split/' + args.dataset + '_valid_setting_' + str(args.setting) + '.csv')
        dfval_drug = list(dfvalid['compound_iso_smiles'])
        dfval_target = list(dfvalid['target_sequence'])
        dfval_y = list(dfvalid['affinity'])

        dftest = pd.read_csv(
            '../GraphDTA/data/' + args.dataset + '/split/' + args.dataset + '_test_setting_' + str(args.setting) + '.csv')
        dftest_drug = list(dftest['compound_iso_smiles'])
        dftest_target = list(dftest['target_sequence'])
        dftest_y = list(dftest['affinity'])
        if 'CNN' in args.penc:
            train_data = ChembertaSeqDataset(xd=dftrain_drug, xt=dftrain_target, y=dftrain_y, embed_ligand=embedded_drug)
            valid_data = ChembertaSeqDataset(xd=dfval_drug, xt=dfval_target, y=dfval_y, embed_ligand=embedded_drug)
            test_data = ChembertaSeqDataset(xd=dftest_drug, xt=dftest_target, y=dftest_y, embed_ligand=embedded_drug)
        elif 'ESM' in args.penc:
            esm_bert, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
            num_esm_layers = len(esm_bert.layers)
            esm_bert.to('cuda:0')
            batch_converter = alphabet.get_batch_converter()

            allseq = set(dftrain_target + dfval_target + dftest_target)

            esmdict={}

            print('Preprocessing seq: ')
            for item in tqdm(allseq):
                if len(item) > 1024:
                    shortitem = item[:1024]
                else:
                    shortitem = item
                _, _, batch_tokens = batch_converter([("", shortitem)])
                with torch.no_grad():
                    results = esm_bert(batch_tokens.to('cuda:0'), repr_layers=[num_esm_layers], return_contacts=True)
                token_representations = results["representations"][num_esm_layers]
                esmdict[item] = token_representations[0, 1: len(shortitem) + 1, :].mean(0).detach().cpu().numpy()

            train_data = ChembertaESMDataset(xd=dftrain_drug,xt=dftrain_target,y=dftrain_y,ESMdict=esmdict,embed_ligand=embedded_drug)
            valid_data = ChembertaESMDataset(xd=dfval_drug,xt=dfval_target,y=dfval_y,ESMdict=esmdict,embed_ligand=embedded_drug)
            test_data = ChembertaESMDataset(xd=dftest_drug,xt=dftest_target,y=dftest_y,ESMdict=esmdict,embed_ligand=embedded_drug)
    else:
        raise NotImplementedError

    return train_data, valid_data, test_data

def embed_ligand(smiles, chem_tokenizer, lig_embedder):
    if len(smiles) > 512:
        smiles = smiles[:512]
    tokens = chem_tokenizer.tokenize(smiles, False)
    input_ligand = torch.LongTensor([tokens['input_ids']])
    try:
        output = lig_embedder(input_ligand, return_dict=True)
    except:
        print(smiles)
        raise AssertionError
    return torch.mean(output.last_hidden_state[0], axis=0).cpu().detach().numpy()

def build_cci_dataset(args):
    if 'GIN' in args.denc:
        print('Batch size: ', args.batch_size)
        print('LR: ', args.lr)

        with open('data/CCI/smile_graph.pkl', 'rb') as handle:
            smile_graph = pickle.load(handle)
        df_train = pd.read_csv('data/CCI/split/train_CCI.csv')
        train_smi1, train_smi2, train_y = list(df_train['smiles1']), list(df_train['smiles2']), list(
            df_train['text_mining'])
        df_test = pd.read_csv('data/CCI/split/val_CCI.csv')
        test_smi1, test_smi2, test_y = list(df_test['smiles1']), list(df_test['smiles2']), list(df_test['text_mining'])

        train_data = CCIDataset(root='data', dataset='CCI', xc1=train_smi1, xc2=train_smi2, y=train_y,
                                smile_graph=smile_graph)
        test_data = CCIDataset(root='data', dataset='CCI', xc1=test_smi1, xc2=test_smi2, y=test_y,
                               smile_graph=smile_graph)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size= args.batch_size,
                                                   collate_fn=collate,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size= args.batch_size, shuffle=False,
                                                  collate_fn=collate,
                                                  drop_last=False)
    elif 'Chemberta' in args.denc:
        print('Batch size: ', args.batch_size)
        print('LR: ', args.lr)
        # Main program: iterate over different datasets

        compound_iso_smiles = []
        df_train = pd.read_csv('data/CCI/split/train_CCI.csv')
        df_test = pd.read_csv('data/CCI/split/train_CCI.csv')
        compound_iso_smiles += list(df_train['smiles2'])
        compound_iso_smiles += list(df_test['smiles1'])
        compound_iso_smiles += list(df_test['smiles2'])
        compound_iso_smiles = set(compound_iso_smiles)
        print(df_train['label'].value_counts())
        print(df_test['label'].value_counts())

        embedded_drug = {}
        all_compound = compound_iso_smiles
        chem_tokenizer = LigandTokenizer()
        lig_embedder = AutoModel.from_pretrained("./data/PubChem10M_SMILES_BPE_450k/")
        lig_embedder.eval()
        print('Getting embedding SMILES feature')
        for smiles in tqdm(all_compound):
            embedded_drug[smiles] = embed_ligand(smiles, chem_tokenizer, lig_embedder)

        train_smi1, train_smi2, train_y = list(df_train['smiles1']), list(df_train['smiles2']), list(df_train['label'])
        test_smi1, test_smi2, test_y = list(df_test['smiles1']), list(df_test['smiles2']), list(df_test['label'])

        train_data = CCIChembertaDataset(xd1=train_smi1, xd2=train_smi2, y=train_y, embed_ligand=embedded_drug)
        test_data = CCIChembertaDataset(xd1=test_smi1, xd2=test_smi2, y=test_y, embed_ligand=embedded_drug)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=False)
    else:
        raise NotImplementedError
    return train_loader, test_loader

def build_PPI_dataset(args):
    modelesm, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    num_esm_layers = len(modelesm.layers)
    modelesm.to("cuda:0")
    batch_converter = alphabet.get_batch_converter()
    traindf = pd.read_csv('data/PPI/split/train_PPI.csv')
    valdf = pd.read_csv('data/PPI/split/val_PPI.csv')

    esmdict = {}
    print('Generating ESM feature')
    listallp = set(list(traindf['prot1_seq']) + list(traindf['prot2_seq']) + list(valdf['prot1_seq']))
    for item in tqdm(listallp):
        if len(item) > 1024:
            shortitem = item[:1024]
        else:
            shortitem = item
        _, _, batch_tokens = batch_converter([("", shortitem)])
        with torch.no_grad():
            results = modelesm(batch_tokens.to("cuda:0"), repr_layers=[num_esm_layers], return_contacts=True)
        token_representations = results["representations"][num_esm_layers]
        esmdict[item] = token_representations[0, 1: len(shortitem) + 1, :].mean(0).detach().cpu().numpy()

    print('Train len: ', len(traindf))
    print('Val len: ', len(valdf))
    print('Test len: ', len(valdf))
    x1 = traindf['prot1_seq']
    x2 = traindf['prot2_seq']
    label = traindf['label']
    traindata = ESMPPIDataset(list(x1), list(x2), list(label), esmdict)

    x1 = valdf['prot1_seq']
    x2 = valdf['prot2_seq']
    label = valdf['label']
    valdata = ESMPPIDataset(list(x1), list(x2), list(label), esmdict)

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True,
                                               drop_last=False, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False,
                                             drop_last=False, num_workers=1)
    return train_loader, val_loader