#!/usr/bin/python3
import json
import time
import numpy as np
from typing import Any
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from wsl.locations import wsl_model_dir
from wsl.loaders.class_loaders import Loader
import torch
from torch.utils.data import DataLoader

model: str = 'rsna_Pneumonia_lr1e-05_bs64_adam_resnet50_first'
debug: bool = True
data: str = 'siim'
column: str = 'Pneumothorax'
extension: str = 'dcm'
classes: int = 1

# +
# def main(model: str = 'rsna_Pneumonia_lr1e-05_bs64_adam_resnet50_first', debug: bool = True, data: str = 'siim', column: str = 'Pneumothorax', extension: str = 'dcm', classes: int = 1):
path = wsl_model_dir / model
print(f'Model: {path}')

if (path / 'configs.json').exists():  # Model not completed
    with open(path / 'configs.json') as f:
        configs = json.load(f)
        # print(configs)
else:
    print('Incomplete model')
    # return

# ------------------------------------------------------
train_dataset = Loader(data=configs['data'],
                       split='train',
                       extension=configs['extension'],
                       classes=configs['classes'],
                       column=configs['column'],
                       regression=configs['regression'],
                       debug=debug)
train_loader = DataLoader(  # type: ignore
    train_dataset, batch_size=configs['batchsize'], num_workers=4,
    pin_memory=True, shuffle=True)

valid_dataset = Loader(data=configs['data'],
                       split='valid',
                       extension=configs['extension'],
                       classes=configs['classes'],
                       column=configs['column'],
                       regression=configs['regression'],
                       debug=debug)
valid_loader = DataLoader(  # type: ignore
    valid_dataset, batch_size=configs['batchsize'], num_workers=4,
    pin_memory=True, shuffle=True)

out_dataset = Loader(data=data,
                     split='valid',
                     extension=extension,
                     classes=classes,
                     column=column,
                     regression=False,
                     debug=debug)
out_loader = DataLoader(  # type: ignore
    out_dataset, batch_size=configs['batchsize'], num_workers=4,
    pin_memory=True, shuffle=True)

print('Length of datasets: In', len(valid_dataset), ' Out', len(out_dataset))

checkpoint = torch.load(path / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint['model'] = checkpoint['model'].module
checkpoint['model'].network = configs['network']
checkpoint['model'].get_map = False
checkpoint['model'].eval()
# sigmoid = torch.nn.Sigmoid()
group_lasso = EmpiricalCovariance(assume_centered=False)
layer_names = {}

# ------------------------------------------------------
def get_mean_precision(loader):

    print('building hook function...')
    features = {}

    def hook(layer, inp, out):
        name = layer_names[layer]
        if name not in features:
            features[name] = out.detach().data.view(out.size(0), out.size(1), -1).mean(dim=-1)
        else:
            features[name] = torch.cat((features[name], out.detach().data.view(out.size(0), out.size(1), -1).mean(dim=-1)), dim=0)
    handles = checkpoint['model'].register_forward_hooks(checkpoint['model'], hook, layer_names)

    start = time.time()
    with torch.set_grad_enabled(False):
        for idx, data in enumerate(loader):
            imgs = data[1].cuda().float()
            _ = checkpoint['model'](imgs)
            speed = configs['batchsize'] * idx // (time.time() - start)
            print('Iter:', idx, 'Speed:', int(speed), 'img/s', end='\r', flush=True)
            if idx > 20:
                break
    print('Total time:', time.time() - start, 'secs')

    print('calculating sample mean...')
    mean = {}
    precision = {}
    for key, value in features.items():
        mean[key] = value.mean(dim=0)
        features[key] -= mean[key]
        group_lasso.fit(features[key].cpu().numpy())
        precision[key] = torch.from_numpy(group_lasso.precision_).float().cuda()

    for handle in handles:
        handle.remove()
    return mean, precision

train_mean, train_precision = get_mean_precision(train_loader)

# ------------------------------------------------------
def get_mahalanobis_score(loader: Any, features: Any, magnitude: float):

    scores = {}
    gaussian = {}
    for layer, name in layer_names.items():
        if 'pool' in name or 'relu' in name or 'bn' in name:
            continue
        checkpoint['optimizer'].zero_grad()

        def hook(layer, inp, out):
            zero_feat = out.view(out.size(0), out.size(1), -1).mean(dim=-1) - train_mean[name]
            gaussian[name] = -0.5 * torch.mm(torch.mm(zero_feat, train_precision[name]), zero_feat.t()).diag()

        handle = layer.register_forward_hook(hook)

        start = time.time()
        for idx, data in enumerate(loader):
            with torch.set_grad_enabled(True):
                imgs = data[1].cuda().float()
                imgs.requires_grad = True
                _ = checkpoint['model'](imgs)

                loss = gaussian[name].mean()
                loss.backward()

                gradient = torch.ge(imgs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2

            with torch.set_grad_enabled(False):
                noisy_imgs = torch.add(imgs.data, gradient, alpha=-magnitude)
                _ = checkpoint['model'](noisy_imgs)
                if name not in scores:
                    scores[name] = gaussian[name].detach().data
                else:
                    scores[name] = torch.cat((scores[name], gaussian[name].detach().data), dim=0)

            checkpoint['optimizer'].zero_grad()
            speed = configs['batchsize'] * idx // (time.time() - start)
            print(name, 'Iter:', idx, 'Speed:', int(speed), 'img/s', end='\r', flush=True)

        scores[name] = scores[name].cpu().numpy()
        handle.remove()
        print()
    return scores

def dict_to_numpy(scores):
    scores_list = []
    for key, value in scores.items():
        scores_list.append(value.tolist())
    scores = np.stack(scores_list)
    return scores.T
    

print('get mahalanobis scores...')
magnitudes = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
maha_in = {}
maha_out = {}
for magnitude in magnitudes:
    print('Noise:', magnitude)
    print('Data - Assumed negative class:', configs['data'])
    in_scores = get_mahalanobis_score(valid_loader, layer_names, magnitude)
    in_scores = dict_to_numpy(in_scores)
    print('Data - Assumed positive class:', data)
    out_scores = get_mahalanobis_score(out_loader, layer_names, magnitude)
    out_scores = dict_to_numpy(out_scores)
    
    X = np.concatenate((in_scores, out_scores), axis=0)
    Y = np.asarray([0] * len(in_scores) + [1] * len(out_scores))
    print(X.shape, Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

    lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
    Y_pred = lr.predict_proba(X_test)[:, 1]
    performance = roc_auc_score(Y_test, Y_pred)
    print(performance)
    print()
# -

Y_pred


