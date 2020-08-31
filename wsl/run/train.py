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
from wsl.loaders.loaders import BinaryLoader


def main(debug: bool = False,
         data: str = 'rsna',
         dicom: bool = True,
         classes: int = 1,
         network: str = 'densenet',
         depth: int = 121,
         wildcat: bool = False,
         pretrained: bool = True,
         optim: str = 'adam',
         resume: bool = False,
         name: str = '',
         lr: float = 1e-6,
         batchsize: int = 64,
         workers: int = 4,
         patience: int = 16,
         balanced: bool = True,
         maps: int = 4,
         alpha: float = 0.0,
         k: int = 1):

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

        full_mname = (data +
                      ('_dicom' if dicom else '') + '_' +
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
    train_dataset = BinaryLoader(data, 'train', 'dcm', debug)
    train_loader = DataLoader(  # type: ignore
        train_dataset, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True
    )

    test_dataset = BinaryLoader(data, 'valid', 'dcm', debug)
    test_loader = DataLoader(  # type: ignore
        test_dataset, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True
    )
    print('done')

    # ------------------------------------------------------
    print('Initializing optim/criterion...', end='')
    if balanced:
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

    # ------------------------------------------------------
    epoch = 0

    best_epoch = 0
    best_loss = 100

    train_loss_all = []
    test_loss_all = []

    train_roc_all = []
    test_roc_all = []

    # ------------------------------------------------------

    while (epoch - best_epoch <= patience):
        start = time.time()
        epoch += 1
        print('Epoch:', epoch, '-Training')
        model.train()
        loss, roc = engine(epoch, train_loader, checkpoint, batchsize, is_train=True)
        train_loss_all.append(loss)
        train_roc_all.append(roc)

        print('Epoch:', epoch, '-Testing')
        model.eval()
        loss, roc = engine(epoch, test_loader, checkpoint, batchsize, is_train=False)
        test_loss_all.append(loss)
        test_roc_all.append(roc)

        os.makedirs(model_dir, exist_ok=True)
        torch.save(checkpoint, model_dir / 'current.pt')

        if best_loss > loss:
            print('Best model updated')
            loss = best_loss
            torch.save(checkpoint, model_dir / 'best.pt')
        else:
            print('Best model unchanged- Epoch:', best_epoch, 'Loss:', best_loss)

        plt.figure(figsize=(12, 18))
        plt.subplot(2, 1, 1)
        plt.plot(train_loss_all, label='Train loss')
        plt.plot(test_loss_all, label='Valid loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(2, 1, 2)
        plt.plot(train_roc_all, label='Train auROC')
        plt.plot(test_roc_all, label='Test auROC')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('auROC')
        plt.savefig(model_dir / 'graphs.png', dpi=300)
        plt.close()

        print('Time taken:', int((time.time() - start) / 60), 'secs')

        if debug:
            print('Breaking early since we are in debug mode')
            print('You can find the trained model at -', model_dir)
            break

    return
