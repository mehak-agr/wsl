#!/usr/bin/python3
import os
import time
import json
import requests
import datetime
import matplotlib.pyplot as plt

from wsl.locations import wsl_model_dir

import torch
from torch import nn
from torch.utils.data import DataLoader

from wsl.networks.medinet.architecture import Architecture
from wsl.networks.medinet.engine import engine
from wsl.loaders.class_loaders import Loader


def main(debug: bool,
         data: str,
         column: str,
         extension: str,
         classes: int,
         augmentation: bool,
         network: str,
         depth: int,
         wildcat: bool,
         pretrained: bool,
         optim: str,
         resume: bool,
         name: str,
         lr: float,
         batchsize: int,
         workers: int,
         patience: int,
         balanced: bool,
         maps: int,
         alpha: float,
         variable_type: str,
         error_range: int,
         ID: str):

    # ------------------------------------------------------
    print('Initializing model...', end='')
    if resume:
        assert len(wsl_model_dir.glob(f'*{name}')) == 1
        full_mname = wsl_model_dir.glob(f'*{name}')[0]
        mname = str(full_mname).split('_')[-1]
    else:
        if debug:
            mname = 'debug'

        elif ID == 'placeholder':
            try:
                # Get a random word to use as a more readable name
                response = requests.get("https://random-word-api.herokuapp.com/word")
                assert response.status_code == 200
                mname = response.json()[0]
            except Exception:
                # As a fallback use the date and time
                mname = datetime.datetime.now().strftime('%d_%m_%H_%M_%S')

        else:
            mname = ID

        full_mname = (data + '_' + column + '_' +
                      f'lr{lr}_bs{batchsize}_{optim}' +
                      ('_pre' if pretrained else '') +
                      ('_bal' if balanced else '') + '_' +
                      f'{network}{depth}' +
                      (f'_wildcat_maps{maps}_alpha{alpha}' if wildcat else '') + '_' +
                      mname)

        model_dir = wsl_model_dir / full_mname
    print('done')
    print('Model Name:', mname)

    # ------------------------------------------------------
    print('Initializing loaders...', end='', flush=True)
    print('train...', end='', flush=True)
    train_dataset = Loader(data,
                           split='train',
                           extension=extension,
                           classes=classes,
                           column=column,
                           variable_type=variable_type,
                           augmentation=augmentation,
                           debug=debug)
    train_loader = DataLoader(  # type: ignore
        train_dataset, batch_size=batchsize, num_workers=workers, shuffle=True
    )

    print('test...', end='', flush=True)
    test_dataset = Loader(data,
                          split='valid',
                          extension=extension,
                          classes=classes,
                          column=column,
                          variable_type=variable_type,
                          debug=debug)
    test_loader = DataLoader(  # type: ignore
        test_dataset, batch_size=batchsize, num_workers=workers, shuffle=True
    )
    print('done')

    # ------------------------------------------------------
    print('Initializing optim/criterion...', end='')
    if resume:
        checkpoint = torch.load(full_mname / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        if variable_type == 'continous':
            criterion = nn.MSELoss()
        elif variable_type == 'categorical':
            criterion = nn.CrossEntropyLoss()
        elif variable_type == 'binary':
            if balanced:
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(train_dataset.pos_weight))
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Variable type should be one of binary/categorical/continous.')
        criterion = criterion.cuda()

        model = Architecture(network, depth, wildcat, classes, maps, alpha, pretrained)
        model = nn.DataParallel(model).cuda()

        if optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1, weight_decay=1e-4)
        elif optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        else:
            raise ValueError(f'{optim} is not supported')

        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'epoch': 0,
            'loss': 100,
            'train_loss_all': [],
            'test_loss_all': [],
            'train_rmetric_all': [],
            'test_rmetric_all': []
        }

    best_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    print('done')
    # ------------------------------------------------------

    while (checkpoint['epoch'] - best_epoch <= patience) and checkpoint['epoch'] < 150:
        start = time.time()
        checkpoint['epoch'] += 1
        print('Epoch:', checkpoint['epoch'], '-Training')
        checkpoint['model'].train()
        checkpoint['loss'], rmetric, summary_train = engine(train_loader, checkpoint, batchsize,
                                                            classes, variable_type, error_range, is_train=True)
        checkpoint['train_loss_all'].append(checkpoint['loss'])
        checkpoint['train_rmetric_all'].append(rmetric)

        print('Epoch:', checkpoint['epoch'], '-Testing')
        checkpoint['model'].eval()
        checkpoint['loss'], rmetric, summary_test = engine(test_loader, checkpoint, batchsize,
                                                           classes, variable_type, error_range, is_train=False)
        checkpoint['test_loss_all'].append(checkpoint['loss'])
        checkpoint['test_rmetric_all'].append(rmetric)

        os.makedirs(model_dir, exist_ok=True)
        torch.save(checkpoint, model_dir / 'current.pt')

        if best_loss > checkpoint['loss']:
            print('Best model updated')
            best_loss = checkpoint['loss']
            best_epoch = checkpoint['epoch']
            torch.save(checkpoint, model_dir / 'best.pt')
        else:
            print('Best model unchanged- Epoch:', best_epoch, 'Loss:', best_loss)

        with open(model_dir / 'summary.txt', 'a+') as file:
            epoch = checkpoint['epoch']
            file.write(f'Epoch: {epoch} \n Train:{summary_train} \n Test:{summary_test}')

        plt.figure(figsize=(12, 18))
        plt.subplot(2, 1, 1)
        plt.plot(checkpoint['train_loss_all'], label='Train loss')
        plt.plot(checkpoint['test_loss_all'], label='Valid loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(2, 1, 2)
        plt.plot(checkpoint['train_rmetric_all'], label='Train rmetric')
        plt.plot(checkpoint['test_rmetric_all'], label='Test rmetric')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('rmetric')
        plt.savefig(model_dir / 'graphs.png', dpi=300)
        plt.close()

        print('Time taken:', int(time.time() - start), 'secs')

        if debug:
            print('Breaking early since we are in debug mode')
            print('You can find the trained model at -', model_dir)
            break

    configs = {
        'name': mname,
        'time': datetime.datetime.now().strftime('%d_%m_%H_%M_%S'),
        'data': data,
        'column': column,
        'extension': extension,
        'classes': classes,
        'network': network,
        'depth': depth,
        'wildcat': wildcat,
        'pretrained': pretrained,
        'optim': optim,
        'learning_rate': lr,
        'batchsize': batchsize,
        'balanced': balanced,
        'maps': maps if wildcat else None,
        'alpha': alpha if wildcat else None,
        'variable_type': variable_type,
        'error_range': error_range if variable_type == 'continous' else None,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'rmetric': checkpoint['test_rmetric_all'][best_epoch - 1],
    }
    with open(model_dir / 'configs.json', 'w') as fp:
        json.dump(configs, fp)
    return
