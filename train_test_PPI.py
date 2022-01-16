import torch
import torch.distributed as dist
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from build_dataset import build_PPI_dataset
from build_model import build_PPI_model
from config import *
from utils import *


def evaluate(all_preds, all_targets):
    metrics = {}
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    aucs = []
    auprs = []

    thresh_metrics = {'F1': {}, 'Precision': {}, 'Recall': {}}
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for metric in thresh_metrics:
        for threshold in THRESHOLDS:
            thresh_metrics[metric][threshold] = []
            thresh_metrics[metric][threshold] = []
            thresh_metrics[metric][threshold] = []

    sorted_preds, sorted_targs = zip(*sorted(zip(all_preds, all_targets), reverse=True))

    num_zero_targets = len(all_targets[all_targets > 0])
    num_one_targets = len(all_targets[all_targets == 0])
    if num_zero_targets > 1 and num_one_targets > 1:
        fpr, tpr, thresh = roc_curve(all_targets, all_preds, pos_label=1)
        p1_auc = auc(fpr, tpr)
        if not math.isnan(p1_auc):
            aucs.append(p1_auc)

    precision, recall, thresh = precision_recall_curve(all_targets, all_preds)
    p1_aupr = auc(recall, precision)
    if not math.isnan(p1_aupr):
        auprs.append(p1_aupr)

    for threshold in THRESHOLDS:
        f1 = compute_f1_score(all_targets, np.copy(all_preds), threshold)
        precision = compute_precision_score(sorted_targs, np.copy(sorted_preds), threshold)
        recall = compute_recall_score(all_targets, np.copy(all_preds), threshold)
        thresh_metrics['F1'][threshold].append(f1)
        thresh_metrics['Precision'][threshold].append(precision)
        thresh_metrics['Recall'][threshold].append(recall)

    acc = (np.round(all_preds) == all_targets).sum() / len(all_preds)

    metrics['acc'] = acc
    metrics['auc'] = np.array(aucs).mean()
    metrics['aupr'] = np.array(auprs).mean()

    return metrics


# training function at each epoch
def train(model, device, train_loader, optimizer, loss_fn):
    model.train()
    totalloss = 0
    all_preds = []
    all_targets = []
    for batch_idx, data in enumerate(train_loader):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        optimizer.zero_grad()
        prediction = model(data1, data2)

        pred = torch.sigmoid(prediction)
        pred = pred.view(-1).detach().cpu()
        target_out = data[0].y.view(-1).detach().cpu()
        all_preds += pred.tolist()
        all_targets += target_out.tolist()

        loss = loss_fn(prediction, data[0].y.to(device))
        loss.backward()
        optimizer.step()
        totalloss += loss.item()
    metric = evaluate(all_preds=all_preds, all_targets=all_targets)
    return totalloss / (batch_idx + 1), metric


def eval(model, device, loader, loss_fn):
    model.eval()
    totalloss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            prediction = model(data1, data2)

            pred = torch.sigmoid(prediction)
            pred = pred.view(-1).detach().cpu()
            target_out = data[0].y.view(-1).detach().cpu()
            all_preds += pred.tolist()
            all_targets += target_out.tolist()

            loss = loss_fn(prediction, data[0].y.to(device))
            totalloss += loss.item()
    metric = evaluate(all_preds=all_preds, all_targets=all_targets)
    return totalloss / (batch_idx + 1), metric


device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
model = build_PPI_model(args).to(device)
train_loader, test_loader = build_PPI_dataset(args)

writer = SummaryWriter("log_PPI/PPI/")
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best_acc = 0
best_epoch = -1
model_file_name = 'results/PPI/PPI_' + args.denc + '.model'
trag = trange(args.epochs)
for epoch in trag:
    trainloss, trainmetric = train(model, device, train_loader, optimizer, loss_fn)
    dist.barrier()
    validloss, valmetric = eval(model, device, test_loader, loss_fn)

    if valmetric['auc'] > best_acc:
        best_acc = valmetric['auc']
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        torch.save(model.state_dict(), model_file_name)
    writer.add_scalar('valid loss', validloss, global_step=epoch)
    writer.add_scalar('valid acc', valmetric['acc'], global_step=epoch)
    writer.add_scalar('valid auc', valmetric['auc'], global_step=epoch)
    writer.add_scalar('train loss', trainloss, global_step=epoch)
    writer.add_scalar('train acc', trainmetric['acc'], global_step=epoch)
    writer.add_scalar('train acc', trainmetric['acc'], global_step=epoch)
    trag.set_postfix(tl=trainloss, vall=validloss, valac=valmetric['acc'],
                     valauc=valmetric['auc'], bestep=best_epoch)

validloss, valmetric = eval(model, device, test_loader, loss_fn)
print(valmetric)
