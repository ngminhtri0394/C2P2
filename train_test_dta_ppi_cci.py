import torch


import random

import torch.nn as nn
from utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from tqdm import tqdm
import numpy as np
import os


from build_model import build_dta_model
from build_dataset import build_dta_dataset
from build_data_loader import build_data_loader
from config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

# training function at each epoch

def unpackdata(data,device):
    if 'GIN' in args.denc:
        drug = data.to(device)
        target = data.target.to(device)
        y = data.y.view(-1).float().to(device)
    elif 'Chemberta' in args.denc:
        drug = data[0].to(device)
        target = data[1].to(device)
        y = data[2].float().to(device)
    else:
        drug = None
        target = None
        y = None
    return drug, target, y

def train(model, device, train_loader, optimizer, loss_fn):
    model.train()
    totalloss = 0
    for batch_idx, data in enumerate(train_loader):
        drug,target,y = unpackdata(data,device)
        optimizer.zero_grad()
        output = model(drug,target)
        loss = loss_fn(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        totalloss += loss.item()
    return totalloss / (batch_idx + 1)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            drug,target,y = unpackdata(data,device)
            output =model(drug,target)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def get_number_of_param(model):
    total_params = 0
    for name, parameter in list(model.named_parameters()):
        param = parameter.numel()
        total_params += param
    return total_params


def build_file_path(args):
    if args.freeze_de:
        strfreeze = 'freeze_de'
    else:
        strfreeze = 'no_freeze_de'
    if args.ccipretrain:
        ccipretrain = '_CCI'
    else:
        ccipretrain = ''
    if args.ppipretrain:
        ppipretrain = '_PPI'
    else:
        ppipretrain=''
    if args.pretrainprojection:
        ptproject = '_pretrainprojection'
    else:
        ptproject = ''
    if args.infograph:
        infographstr = '_infograph'
    else:
        infographstr = ''
    if args.train:
        shot = ''
    else:
        shot = '_zeroshot'
    return 'b_{}_lr_{}_ds_{}_pe_{}{}{}_de_{}{}{}_st_{}_{}_{}{}'.format(args.batch_size, args.lr, args.dataset, args.penc, ppipretrain,ptproject, args.denc, ccipretrain,infographstr,
                                                          args.setting, strfreeze,args.name,shot)

def adjust_learning_rate(optimizer, LR, scale=0.9):
    """Sets the learning rate to the initial LR decayed by 10 every interval epochs"""
    lr = LR * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_process(model_file_name, basefilepath):
    writer = SummaryWriter("weights/" + basefilepath + "/log/")
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',patience=20,factor=0.9)
    best_mse = 1000
    best_epoch = -1
    trg = trange(args.epochs)
    for epoch in trg:
        trainloss = train(model, device, train_loader, optimizer, loss_fn)
        G, P = predicting(model, device, valid_loader)
        val = mse(G, P)
        writer.add_scalar('train_loss', trainloss, global_step=epoch)
        writer.add_scalar('valid_loss', val, global_step=epoch)
        # G, P = predicting(model, device, test_loader)
        # testmse = mse(G, P)
        # writer.add_scalar('test_loss', testmse, global_step=epoch)
        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
        # scheduler.step(val)
        trg.set_postfix(loss=trainloss, valloss=val, bestloss=best_mse, bestep=best_epoch)
    return best_mse


def test_process(model_file_name, result_file_name):
    model.load_state_dict(torch.load(model_file_name))
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    G, P = predicting(model, device, test_loader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    with open(result_file_name, 'w') as f:
        f.write(','.join(map(str, ret)))


# python train_test_dta_ppi_cci.py --dataset davis --penc ESM --ppipretrain --pretrainprojection --ccipretrain --denc GIN --setting 2 --epochs 1000 --lr 0.001 --notrain --notest

model = build_dta_model(args)
print('Number of param: ', get_number_of_param(model))
train_data, valid_data, test_data = build_dta_dataset(args)
train_loader, valid_loader, test_loader = build_data_loader(args,train_data,valid_data,test_data)
basefilepath = build_file_path(args)

print('Learning rate: ', args.lr)
print('Epochs: ', args.epochs)
print('Setting ', args.setting)

model_file_name = 'weights/' + basefilepath + '/best_model.pt'
result_file_name = 'weights/' + basefilepath + '/result.csv'
if not os.path.isdir('weights/' + basefilepath + '/'):
    os.makedirs('weights/' + basefilepath + '/')
if args.train:
    print('Train')
    best_val_loss = train_process(model_file_name, basefilepath)

if args.test:
    print('Test')
    test_process(model_file_name, result_file_name)
