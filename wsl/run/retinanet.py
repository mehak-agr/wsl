#!/usr/bin/python3
import os
import time
import json
import requests
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from wsl.locations import wsl_model_dir

import torch
from torch import nn
from torch.utils.data import DataLoader

from wsl.networks.retinanet import architecture
from wsl.networks.retinanet.engine import engine, engine_boxes
from wsl.loaders.detect_loaders import Loader


def main(debug: bool,
         data: str,
         column: str,
         extension: str,
         classes: int,
         depth: int,
         pretrained: bool,
         optim: str,
         resume: bool,
         results: bool,
         name: str,
         lr: float,
         batchsize: int,
         workers: int,
         patience: int,
         ID: str):
    
    # ------------------------------------------------------
    if resume:
        matching_models = list(wsl_model_dir.glob(f'*{name}'))
        assert len(matching_models) == 1
        model_dir = matching_models[0]
        mname = str(model_dir).split('_')[-1]
        print(mname)

    elif results:
        if 'retinanet' in name:
            matching_models = list(wsl_model_dir.glob(f'*{name}*/configs.json'))
        else:
            matching_models = list(wsl_model_dir.glob(f'*retinanet*{name}*/configs.json'))
        model_dirs = [model_dir.parent for model_dir in matching_models]

    else:
        print('Initializing model...', end='')
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
                      f'lr{lr}_bs{batchsize}_{optim}' + '_' +
                      ('pre_' if pretrained else '') +
                      f'retinanet{depth}' + '_' +
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
                           debug=debug)
    train_loader = DataLoader(  # type: ignore
        train_dataset, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True
    )

    print('test...', end='', flush=True)
    test_dataset = Loader(data,
                          split='valid',
                          extension=extension,
                          classes=classes,
                          column=column,
                          debug=debug)
    test_loader = DataLoader(  # type: ignore
        test_dataset, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True
    )
    print('done')

    if classes > 1:
        print('Class List: ', train_dataset.class_names)
        
    # ------------------------------------------------------
    
    if results:
        for model_dir in model_dirs:
            print('Initializing optim/checkpoint...')
            checkpoint = torch.load(model_dir / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
            with open(model_dir / 'configs.json') as f:
                configs = json.load(f)
            print('Calculating box results...')
            checkpoint['model'].eval()
            df, rmetric, wild_metric, pointwise = engine_boxes(test_dataset, checkpoint)
            df.to_csv(model_dir / 'results.csv')

            configs['rmetric'] = rmetric
            configs['wild_metric'] = wild_metric
            configs['utility'] = wild_metric
            configs['pointwise'] = pointwise

            with open(model_dir / 'configs.json', 'w') as fp:
                json.dump(configs, fp)
            print('Finished:', rmetric, wild_metric, pointwise)
            print(f'You can find the calculated results at - {model_dir}/results.csv')
        return

    # ------------------------------------------------------
    print('Initializing optim/checkpoint...', end='')
    if resume or results:
        checkpoint = torch.load(model_dir / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        if depth == 18:
            model = architecture.resnet18(classes, pretrained)
        elif depth == 34:
            model = architecture.resnet34(classes, pretrained)
        elif depth == 50:
            model = architecture.resnet50(classes, pretrained)
        elif depth == 101:
            model = architecture.resnet101(classes, pretrained)
        elif depth == 152:
            model = architecture.resnet152(classes, pretrained)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
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
            'epoch': 0,
            'loss': 100,
            'train_loss_all': [],
            'test_loss_all': [],
            'train_class_loss_all': [],
            'test_class_loss_all': [],
            'train_reg_loss_all': [],
            'test_reg_loss_all': []
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
        class_loss, reg_loss, checkpoint['loss'], summary_train = engine(train_loader, checkpoint, batchsize, classes, is_train=True)
        checkpoint['train_loss_all'].append(checkpoint['loss'])
        checkpoint['train_class_loss_all'].append(class_loss)
        checkpoint['train_reg_loss_all'].append(reg_loss)

        print('Epoch:', checkpoint['epoch'], '-Testing')
        # checkpoint['model'].eval() - In eval mode model prints boxes
        class_loss, reg_loss, checkpoint['loss'], summary_test = engine(test_loader, checkpoint, batchsize, classes, is_train=False)
        checkpoint['test_loss_all'].append(checkpoint['loss'])
        checkpoint['test_class_loss_all'].append(class_loss)
        checkpoint['test_reg_loss_all'].append(reg_loss)

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
        plt.subplot(3, 1, 1)
        plt.plot(checkpoint['train_loss_all'], label='Train loss')
        plt.plot(checkpoint['test_loss_all'], label='Valid loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(3, 1, 2)
        plt.plot(checkpoint['train_class_loss_all'], label='Train Class loss')
        plt.plot(checkpoint['test_class_loss_all'], label='Valid Class loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Class Loss'),
        plt.subplot(3, 1, 3)
        plt.plot(checkpoint['train_reg_loss_all'], label='Train Reg loss')
        plt.plot(checkpoint['test_reg_loss_all'], label='Valid Reg loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Reg Loss')
        plt.savefig(model_dir / 'graphs.png', dpi=300)
        plt.close()

        print('Time taken:', int(time.time() - start), 'secs')

        if debug:
            print('Breaking early since we are in debug mode')
            print('You can find the trained model at -', model_dir)
            break

    checkpoint = torch.load(model_dir / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    df, rmetric, wild_metric, pointwise = engine_boxes(test_dataset, checkpoint)
    df.to_csv(model_dir / 'results.csv')
    print('Finished:', rmetric)
    print(f'You can find the calculated results at - {model_dir}/results.csv')
    
    configs = {
        'name': mname,
        'time': datetime.datetime.now().strftime('%d_%m_%H_%M_%S'),
        'data': data,
        'column': column,
        'extension': extension,
        'classes': classes,
        'network': 'retinanet',
        'depth': depth,
        'pretrained': pretrained,
        'optim': optim,
        'learning_rate': lr,
        'batchsize': batchsize,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'rmetric': rmetric,
        'wild_metric': wild_metric,
        'pointwise': pointwise
    }
    with open(model_dir / 'configs.json', 'w') as fp:
        json.dump(configs, fp)
    return
