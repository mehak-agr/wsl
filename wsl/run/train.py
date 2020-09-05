#!/usr/bin/python3
import os
import time
import requests
import datetime
import matplotlib.pyplot as plt

from wsl.locations import wsl_model_dir

import torch
from torch import nn
from torch.utils.data import DataLoader

from wsl.networks.architecture import Architecture
from wsl.networks.engine import engine
from wsl.loaders.loaders import Loader


def main(debug: bool,
         data: str,
         col_name: str,
         extension: str,
         classes: int,
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
         k: int,
         regression: bool,
         error_range: int):

    # ------------------------------------------------------
    print('Initializing model...', end='')
    if resume:
        assert len(wsl_model_dir.glob(f'*{name}')) == 1
        full_mname = wsl_model_dir.glob(f'*{name}')[0]
        model = torch.load(full_mname, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            # Get a random word to use as a more readable name
            response = requests.get("https://random-word-api.herokuapp.com/word")
            assert response.status_code == 200
            mname = response.json()[0]
        except Exception:
            # As a fallback use the date and time
            mname = datetime.datetime.now().strftime('%d_%m_%H_%M_%S')

        full_mname = (('debug_' if debug else '') +
                      data + '_' + col_name + '_' +
                      f'lr{lr}_bs{batchsize}_{optim}' +
                      ('_pre' if pretrained else '') +
                      ('_bal' if balanced else '') + '_' +
                      f'{network}{depth}' +
                      (f'_wildcat_maps{maps}_alpha{alpha}_k{k}' if 'wildcat' else '') + '_' +
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
                           col_name=col_name,
                           regression=regression,
                           debug=debug)
    train_loader = DataLoader(  # type: ignore
        train_dataset, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True
    )

    print('test...', end='', flush=True)
    test_dataset = Loader(data,
                          split='valid',
                          extension=extension,
                          classes=classes,
                          col_name=col_name,
                          regression=regression,
                          debug=debug)
    test_loader = DataLoader(  # type: ignore
        test_dataset, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True
    )
    print('done')

    if regression:
        reg_args = {'max': train_dataset.lmax,
                    'min': train_dataset.lmin,
                    'error_range': error_range}
    else:
        reg_args = None

    if classes > 1:
        print('Class List: ', train_dataset.class_names)

    # ------------------------------------------------------
    print('Initializing optim/criterion...', end='')
    if regression:
        criterion = nn.MSELoss()
    elif balanced:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(train_dataset.pos_weight))
    else:
        criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    model = Architecture(network, depth, wildcat, classes, maps, alpha, k, pretrained)
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
        'criterion': criterion}

    print('done')
    # ------------------------------------------------------
    epoch = 0

    best_epoch = 0
    best_loss = 100

    train_loss_all = []
    test_loss_all = []

    train_rmetric_all = []
    test_rmetric_all = []

    # ------------------------------------------------------

    while (epoch - best_epoch <= patience):
        start = time.time()
        epoch += 1
        print('Epoch:', epoch, '-Training')
        model.train()
        loss, rmetric, summary_train = engine(epoch, train_loader, checkpoint,
                                              batchsize, classes, reg_args, is_train=True)
        train_loss_all.append(loss)
        train_rmetric_all.append(rmetric)

        print('Epoch:', epoch, '-Testing')
        model.eval()
        loss, rmetric, summary_test = engine(epoch, test_loader, checkpoint,
                                             batchsize, classes, reg_args, is_train=False)
        test_loss_all.append(loss)
        test_rmetric_all.append(rmetric)

        os.makedirs(model_dir, exist_ok=True)
        torch.save(checkpoint, model_dir / 'current.pt')

        if best_loss > loss:
            print('Best model updated')
            loss = best_loss
            torch.save(checkpoint, model_dir / 'best.pt')
        else:
            print('Best model unchanged- Epoch:', best_epoch, 'Loss:', best_loss)

        with open(model_dir / 'summary.txt', 'a+') as file:
            file.write(f'Epoch: {epoch} \n Train:{summary_train} \n Test:{summary_test}')

        plt.figure(figsize=(12, 18))
        plt.subplot(2, 1, 1)
        plt.plot(train_loss_all, label='Train loss')
        plt.plot(test_loss_all, label='Valid loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(2, 1, 2)
        plt.plot(train_rmetric_all, label='Train rmetric')
        plt.plot(test_rmetric_all, label='Test rmetric')
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

    return
