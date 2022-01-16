from math import sqrt
import sklearn
import numpy as np
from scipy import stats
import math
from torch_geometric.data import Batch
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        if not ch.isalpha():
            continue
        x[i] = seq_dict[ch]
    return x


def seq_cat_clstoken(prot):
    x = np.zeros(max_seq_len + 1)
    for i, ch in enumerate(prot[:max_seq_len]):
        if not ch.isalpha():
            continue
        x[i] = seq_dict[ch]
    # x[max_seq_len] = seq_dict['/']
    return x


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def compute_precision_score(all_targets, all_preds_copy, thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    if len(all_preds_copy[all_preds_copy == 1]) > 0:
        precision = sklearn.metrics.precision_score(all_targets, all_preds_copy)
    else:
        precision = math.nan
    return precision


def compute_recall_score(all_targets, all_preds_copy, thresh):
    all_preds_copy[all_preds_copy > thresh] = 1
    all_preds_copy[all_preds_copy <= thresh] = 0
    if len(all_preds_copy[all_preds_copy == 1]) > 0:
        recall = sklearn.metrics.recall_score(all_targets, all_preds_copy)
    else:
        recall = math.nan
    return recall


def compute_f1_score(all_targets, all_preds_copy, thresh):
    all_preds_copy[all_preds_copy >= thresh] = 1
    all_preds_copy[all_preds_copy < thresh] = 0
    f1 = sklearn.metrics.f1_score(all_targets, all_preds_copy)
    return f1

def getROCE(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x: x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if (targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if (fp1 > ((roceRate * n) / 100)):
                break
    roce = (tp1 * n) / (p * fp1)
    return roce

def collate(data_list):
    # print("Run collate")
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
