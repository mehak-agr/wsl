#!/usr/bin/python3
# +
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt

import json
import datetime
import cv2
from sklearn.metrics import roc_auc_score

from wsl.locations import wsl_model_dir, wsl_summary_dir
from wsl.loaders.class_loaders import Loader
from wsl.networks.medinet.utils import box_to_map, rle2mask


# -

def main(name: str,
         task: str,
         store: bool):

    if name == 'all':
        models = wsl_model_dir.glob('*')
    else:
        models = wsl_model_dir.glob(f'*{name}*')
    models = list(models)
    print('Number of potential model matches =', len(models))
    all_configs = []
    for m, path in enumerate(models):

        if 'debug' in str(path):  # Debugging model
            print('Debugging model')
            continue
        elif 'wildcat' not in str(path):  # Model is not wildcat
            print('Model is not wildcat')
            continue
        elif not (path / 'configs.json').exists():  # Model not completed
            print('Model not completed')
            continue
        else:
            with open(path / 'configs.json') as f:
                configs = json.load(f)
                print(configs)
            dataset = Loader(data=configs['data'],
                             split='valid',
                             extension=configs['extension'],
                             classes=configs['classes'],
                             column=configs['column'],
                             regression=configs['regression'])

        print(f'Model {m} : {path}')

        checkpoint = torch.load(path / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint['model'] = checkpoint['model'].module
        checkpoint['model'].get_map = True
        checkpoint['model'].eval()

        org_size = (1024, 1024)
        new_size = (224, 224)
        sigmoid = torch.nn.Sigmoid().cuda()
        all_scores = []

        with torch.set_grad_enabled(False):
            for idx, data in enumerate(dataset):
                img, label = data
                name = dataset.names[idx]
                labels = dataset.labels[idx]

                predicted_map = checkpoint['model'](img.unsqueeze(dim=0).cuda().float()).squeeze(dim=0)
                predicted_map = sigmoid(predicted_map.sum(dim=0)).cpu().data.numpy()

                score = []
                for i, label in enumerate(labels):
                    if label == 0:
                        continue

                    if task == 'detect':
                        ground_map = box_to_map(dataset.df[dataset.df.Id == name].box.to_list(),
                                                np.zeros(org_size))
                    elif task == 'segment':
                        ground_map = np.zeros(org_size)
                        eps = dataset.df[dataset.df.Id == name].EncodedPixels.to_list()
                        for ep in eps:
                            ground_map += rle2mask(ep, np.zeros(org_size)).T
                    else:
                        print('Ground truth not available.')

                    # plt.imshow(ground_map)
                    ground_map = cv2.resize(ground_map, new_size, interpolation=cv2.INTER_NEAREST).clip(0, 1)
                    re_pred_map = cv2.resize(predicted_map[i], new_size, interpolation=cv2.INTER_AREA)
                    score.append(roc_auc_score(ground_map.flatten(), re_pred_map.flatten()))

                all_scores += score
                if (len(all_scores) + 1) % 32 == 0:
                    print('Idx:', idx, 'Mean:', np.mean(all_scores), end='\r')

            configs['wild'] = np.mean(all_scores)

        all_configs.append(configs)

    df = pd.DataFrame.from_dict(all_configs)
    print(df)
    time = datetime.datetime.now().strftime('%H_%d_%m')
    if store:
        df.to_csv(wsl_summary_dir / f'wild_{time}')


