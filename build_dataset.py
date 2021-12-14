from dataset.GraphESMFeatureDataset import GraphESMFeatureDataset
from dataset.GraphSeqDataset import GraphSeqDataset
from dataset.ChembertaSeqDataset import ChembertaSeqDataset
from dataset.ChembertaESMDataset import ChembertaESMDataset
import pickle
import pandas as pd
from torch_geometric.data import DataLoader as GMDataLoader
from torch.utils.data import DataLoader as PTDataLoader
import esm
from tqdm import tqdm
import torch

def build_dataset(args):
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
        train_data = None
        valid_data = None
        test_data = None

    return train_data, valid_data, test_data

