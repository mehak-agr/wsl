#!/usr/bin/python3
import time
from typing import Dict, Any
import numpy as np
from wsl.networks.medinet.utils import regression_accuracy
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from monai.metrics import compute_roc_auc, compute_confusion_metric
import torch


def engine(loader: Any, checkpoint: Dict[str, Any], batchsize: int,
           classes: int, variable_type: str, error_range: int, is_train: bool):

    overall_loss = []
    all_preds = torch.zeros((0, classes))
    all_labels = torch.zeros((0, classes))
    labels_onehot = torch.FloatTensor(batchsize, classes).cuda()
    start = time.time()
    sigmoid = torch.nn.Sigmoid()

    with torch.set_grad_enabled(is_train):
        for iter_num, data in enumerate(loader):
            # name = data[0]
            imgs = data[1].cuda().float()
            labels = data[2].cuda()

            predicted = checkpoint['model'](imgs)
            loss = checkpoint['criterion'](predicted, labels)
            predicted, labels = predicted.detach(), labels.detach()

            if is_train:
                loss.backward()
                checkpoint['optimizer'].step()
                checkpoint['optimizer'].zero_grad()

            overall_loss.append(float(loss.item()))
            all_preds = torch.cat((predicted, all_preds))

            if variable_type == 'categorical':
                if labels_onehot.shape[0] != labels.shape[0]:
                    labels_onehot = torch.FloatTensor(labels.shape[0], classes).cuda()
                labels_onehot.zero_()
                labels_onehot.scatter_(1, labels.unsqueeze(dim=1), 1)
                all_labels = torch.cat((labels_onehot.float(), all_labels))
                predicted = predicted.max(dim=1)[1]  # for correct printing
            else:
                all_labels = torch.cat((labels, all_labels))

            speed = batchsize * iter_num // (time.time() - start)
            print('Epoch:', checkpoint['epoch'], 'Iter:', iter_num,
                  'Pred:', round(predicted.float().mean().item(), 3),
                  'Label:', round(labels.float().mean().item(), 3),
                  'Loss:', round(np.mean(overall_loss), 3),
                  'Speed:', int(speed), 'img/s', end='\r', flush=True)

    loss = np.mean(overall_loss)
    if variable_type == 'continous':
        all_labels, all_preds = all_labels.cpu(), all_preds.cpu()
        rmetric = r2_score(all_labels, all_preds)
        acc = regression_accuracy(all_labels, all_preds, error_range)
        spear, pvalue = spearmanr(all_preds, all_labels)
        summary = (f'Epoch Summary - Loss:{round(loss, 3)} Spearman:{round(spear, 2)} PValue:{round(pvalue, 3)} ' +
                   f'R2:{round(rmetric, 1)} Accuracy(at {error_range}):{round(100 * acc, 1)}')

    else:
        rmetric = compute_roc_auc(all_preds, all_labels, other_act=sigmoid)
        sens = compute_confusion_metric(all_preds, all_labels,
                                        activation=sigmoid, metric_name='sensitivity')
        spec = compute_confusion_metric(all_preds, all_labels,
                                        activation=sigmoid, metric_name='specificity')
        summary = (f'Epoch Summary- Loss:{round(loss, 3)}  ROC:{round(rmetric * 100, 1)} ' +
                   f'Sensitivity:{round(100 * sens, 1)}  Specificity: {round(100 * spec, 1)}')

    print(summary)
    return loss, rmetric, summary
