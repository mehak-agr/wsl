#!/usr/bin/python3
import json
import time
from typing import Any
from sklearn.covariance import EmpiricalCovariance

from wsl.locations import wsl_model_dir
from wsl.loaders.class_loaders import Loader
import torch
from torch.utils.data import DataLoader


def main(out_data: str = 'chexpert'):
    models = wsl_model_dir.glob('*')
    # all_configs = []

    for idx, path in enumerate(models):
        if 'debug' in str(path):  # Debugging model
            continue
        elif not (path / 'configs.json').exists():  # Model not completed
            continue
        else:
            with open(path / 'configs.json') as f:
                configs = json.load(f)
                # print(configs)
        print(f'Model {idx} : {path}')

        # ------------------------------------------------------
        train_dataset = Loader(data=configs['data'],
                               split='train',
                               extension=configs['extension'],
                               classes=configs['classes'],
                               column=configs['column'],
                               regression=configs['regression'])
        train_loader = DataLoader(  # type: ignore
            train_dataset, batch_size=configs['batchsize'], num_workers=4,
            pin_memory=True, shuffle=True)

        valid_dataset = Loader(data=configs['data'],
                               split='valid',
                               extension=configs['extension'],
                               classes=configs['classes'],
                               column=configs['column'],
                               regression=configs['regression'])
        valid_loader = DataLoader(  # type: ignore
            valid_dataset, batch_size=configs['batchsize'], num_workers=4,
            pin_memory=True, shuffle=True)

        out_dataset = Loader(data=out_data,
                             split='valid',
                             extension=configs['extension'],
                             classes=configs['classes'],
                             column=configs['column'],
                             regression=configs['regression'])
        out_loader = DataLoader(  # type: ignore
            out_dataset, batch_size=configs['batchsize'], num_workers=4,
            pin_memory=True, shuffle=True)

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
                    imgs = data[0].cuda().float()
                    _ = data[1]
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
                        print(scores[name].mean())

                    checkpoint['optimizer'].zero_grad()
                    speed = configs['batchsize'] * idx // (time.time() - start)
                    print(name, 'Iter:', idx, 'Speed:', int(speed), 'img/s', end='\r', flush=True)

                handle.remove()
                print()
            return scores

        print('get mahalanobis scores...')
        magnitudes = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
        maha_valid_scores = {}
        maha_out_scores = {}
        for magnitude in magnitudes:
            print('Noise:', magnitude)
            print('Data - Assumed negative class:', configs['data'])
            maha_valid_scores[magnitude] = get_mahalanobis_score(valid_loader, layer_names, magnitude)
            print('Data - Assumed positive class:', out_data)
            maha_out_scores[magnitude] = get_mahalanobis_score(out_loader, layer_names, magnitude)
            print()

        print('merge mahalanobis scores...')

