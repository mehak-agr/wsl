#!/usr/bin/python3
import time
from typing import Dict, Any
import numpy as np
from wsl.networks.utils import regression_accuracy
from sklearn.metrics import r2_score
from monai.metrics import compute_roc_auc, compute_confusion_metric
import torch


def engine(loader: Any, checkpoint: Dict[str, Any],
           batchsize: int, classes: int, reg_args: Any, is_train: bool):

    overall_loss = []
    all_preds = torch.zeros((0, classes))
    all_labels = torch.zeros((0, classes))
    start = time.time()
    sigmoid = torch.nn.Sigmoid()

    with torch.set_grad_enabled(is_train):
        for iter_num, data in enumerate(loader):
            imgs = data[0].cuda().float()
            labels = data[1].cuda().float()

            predicted = checkpoint['model'](imgs)
            loss = checkpoint['criterion'](predicted, labels)

            if is_train:
                loss.backward()
                checkpoint['optimizer'].step()
                checkpoint['optimizer'].zero_grad()

            overall_loss.append(float(loss.item()))
            all_preds = torch.cat((predicted.detach(), all_preds))
            all_labels = torch.cat((labels.detach(), all_labels))

            speed = batchsize * iter_num // (time.time() - start)
            print('Epoch:', checkpoint['epoch'], 'Iter:', iter_num,
                  'Running loss:', round(np.mean(overall_loss), 3),
                  'Speed:', int(speed), 'img/s', end='\r', flush=True)

    loss = np.mean(overall_loss)
    if reg_args is None:
        rmetric = compute_roc_auc(all_preds, all_labels, other_act=sigmoid)
        sens = compute_confusion_metric(all_preds, all_labels,
                                        activation=sigmoid, metric_name='sensitivity')
        spec = compute_confusion_metric(all_preds, all_labels,
                                        activation=sigmoid, metric_name='specificity')
        summary = (f'Epoch Summary- Loss:{round(loss, 3)}  ROC:{round(rmetric * 100, 1)} ' +
                   f'Sensitivity:{round(100 * sens, 1)}  Specificity: {round(100 * spec, 1)}')
    else:
        error_range = reg_args['error_range']
        all_labels = [((x * reg_args['max']) + reg_args['min']).item() for x in all_labels]
        all_preds = [((x * reg_args['max']) + reg_args['min']).item() for x in all_preds]
        rmetric = r2_score(all_labels, all_preds)
        a1 = regression_accuracy(all_labels, all_preds, error_range)
        a2 = regression_accuracy(all_labels, all_preds, error_range)
        summary = (f'Epoch Summary- Loss:{round(loss, 3)}  R2:{round(rmetric, 1)} ' +
                   f'Accuracy at {error_range}:{round(100 * a1, 1)} ' +
                   f'Accuracy at {(error_range * 2)}:{round(100 * a2, 1)}')

    print(summary)
    return loss, rmetric, summary
