import torch
from model_dta.DTA import DTA
from model_dta.DTA_transfer import DTA_transfer
from model_dta.ESM_Feat_Encoder import ESM_Feat_Encoder as ESM_Feat_Encoder
from model_dta.ESM1v_Feat_Encoder import ESM1v_Feat_Encoder as ESM1v_Feat_Encoder
from model_dta.ESM1b_Feat_Encoder import ESM1b_Feat_Encoder as ESM1b_Feat_Encoder
from model_dta.ESMMSA_Feat_Encoder import ESMMSA_Feat_Encoder as ESMMSA_Feat_Encoder
from model_dta.GIN_Encoder import GIN_Encoder as GIN_Encoder
from model_dta.ESM_Conv_Feat_Encoder import ESM_Conv_Feat_Encoder
from model_dta.Chemberta_Encoder import Chemberta_Encoder
from model_dta.CNN_Feat_Encoder import CNN_Feat_Encoder
import sys


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def load_pretrain_cci(dencoder):
    state_dict = torch.load('pretrain_weight/CCI/model_CCI_GIN_CCI_dim_128_cls_full_parallel.model')
    for key in list(state_dict.keys()):
        # state_dict[key.replace('module.', '')] = state_dict.pop(key)
        layername = key.split('.')[1]
        try:
            layer_idx = int(layername[-1:])-1
        except:
            layer_idx = layername
        layertype = layername[:-1]
        newname = 'layers.'+str(layer_idx)+'.'+layertype
        state_dict[key.replace('module.'+layername, newname)] = state_dict.pop(key)
    dencoder.load_state_dict(state_dict,strict=False)
    return dencoder

def load_pretrain_Chemberta_CCI(dencoder):
    state_dict = torch.load('pretrain_weight/CCI/model_CCI_Chemberta_CCI_dim_0_cls_chmberta.model')
    dencoder.load_state_dict(state_dict,strict=False)
    return dencoder

def load_pretrain_ppi(args, pencoder):
    state_dict = torch.load('pretrain_weight/PPI/model_'+args.penc+'_PPI_String.model')
    for key in list(state_dict.keys()):
        print(key)
        state_dict[key.replace('module.', '')] = state_dict.pop(key)
    pencoder.load_state_dict(state_dict,strict=False)
    return pencoder

def load_pretrain_infograph(dencoder):
    state_dict = torch.load('pretrain_weight/Infograph/bestmodel.pt')
    for key in list(state_dict.keys()):
        print(key)
        state_dict[key.replace('encoder.', '')] = state_dict.pop(key)
    dencoder.load_state_dict(state_dict,strict=False)
    return dencoder


def build_model(args):
    pencoder = getattr(sys.modules[__name__], args.penc+'_Feat_Encoder')(indim=args.esmdim)
    if args.ppipretrain:
        print('Load pretrain PPI')
        pencoder = load_pretrain_ppi(args,pencoder)
    dencoder = getattr(sys.modules[__name__], args.denc+ '_Encoder')(outdim=args.dencdim)
    if args.ccipretrain:
        print('Load pretrain CCI')
        if args.denc == 'GIN':
            dencoder = load_pretrain_cci(dencoder)
            if args.freeze_de:
                print('Freeze drug encoder')
                for name, param in list(dencoder.named_parameters()):
                    param.requires_grad = False
        elif args.denc == 'Chemberta':
            dencoder = load_pretrain_Chemberta_CCI(dencoder)
    elif args.infograph:
        print('Load pretrain info graph')
        if args.denc == 'GIN':
            dencoder = load_pretrain_infograph(dencoder)
    model = DTA(pencoder=pencoder, dencoder=dencoder,poutdim=args.pencdim,doutdim=args.dencdim)
    return model

def build_model_check_transfer(args):
    pencoder = getattr(sys.modules[__name__], args.penc+'_Feat_Encoder')()
    print('before load weight')
    print(list(pencoder.named_parameters())[0][0])
    if args.ppipretrain:
        print('Load pretrain PPI')
        pencoder = load_pretrain_ppi(args,pencoder)
    print('after load weight')
    print(list(pencoder.named_parameters())[0][0])
    dencoder = getattr(sys.modules[__name__], args.denc+ '_Encoder')(outdim=args.dencdim)
    if args.ccipretrain:
        print('Load pretrain CCI')
        if args.denc == 'GIN':
            dencoder = load_pretrain_cci(dencoder)
            if args.freeze_de:
                print('Freeze drug encoder')
                for name, param in list(dencoder.named_parameters()):
                    param.requires_grad = False
        elif args.denc == 'Chemberta':
            dencoder = load_pretrain_Chemberta_CCI(dencoder)
    elif args.infograph:
        print('Load pretrain info graph')
        if args.denc == 'GIN':
            dencoder = load_pretrain_infograph(dencoder)
    model = DTA_transfer(pencoder=pencoder, dencoder=dencoder,poutdim=args.pencdim,doutdim=args.dencdim)
    return model