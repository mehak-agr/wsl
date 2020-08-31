# +
import time
from typing import Dict, Any
import numpy as np
from monai.metrics import compute_roc_auc, compute_confusion_metric

import torch


def engine(epoch: int, loader: Any, checkpoint: Dict[str, Any], batchsize: int, is_train: bool):
    print('Initializing engine...')
    overall_loss = []
    all_preds = torch.zeros((0, 1))
    all_labels = torch.zeros((0, 1))
    start = time.time()
    sigmoid = torch.nn.Sigmoid()

    with torch.set_grad_enabled(is_train):
        for iter_num, data in enumerate(loader):
            imgs = data[0].cuda().float()
            labels = data[1].unsqueeze(dim=-1).cuda().float()

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
            print('Epoch:', epoch, 'Iter:', iter_num,
                  'Running loss:', round(np.mean(overall_loss), 3),
                  'Speed:', int(speed), 'img/s', end='\r', flush=True)

    loss = np.mean(overall_loss)
    roc = compute_roc_auc(all_preds, all_labels, other_act=sigmoid)
    sens = compute_confusion_metric(all_preds, all_labels, activation=sigmoid, metric_name='sensitivity')
    spec = compute_confusion_metric(all_preds, all_labels, activation=sigmoid, metric_name='specificity')
    print('Epoch Summary- Loss:', round(loss, 3), 'ROC:', round(roc * 100, 1),
          'Sensitivity:', round(100 * sens, 1), 'Specificity', round(100 * spec, 1))

    return loss, roc
