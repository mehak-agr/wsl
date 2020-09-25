#!/usr/bin/python3
# +
import torch
import numpy as np
import pandas as pd

import json
import datetime
import cv2
from sklearn.metrics import roc_auc_score

from wsl.locations import wsl_model_dir, wsl_summary_dir
from wsl.loaders.class_loaders import Loader
from wsl.networks.medinet.utils import box_to_map


# -

def main(store: bool = False):
    models = wsl_model_dir.glob('*')
    all_configs = []
    for idx, path in enumerate(models):

        if 'debug' in str(path):  # Debugging model
            continue
        elif 'wildcat' not in str(path):  # Model is not wildcat
            continue
        elif not (path / 'configs.json').exists():  # Model not completed
            continue
        else:
            with open(path / 'configs.json') as f:
                configs = json.load(f)
            dataset = Loader(data=configs['data'],
                             split='valid',
                             extension=configs['extension'],
                             classes=configs['classes'],
                             column=configs['column'],
                             regression=configs['regression'])
            print(configs)

        print(f'Model {idx} : {path}')

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

                    ground_map = box_to_map(dataset.df[dataset.df.Id == name].to_dict(orient='row'),
                                            configs['column'], org_size, new_size)
                    re_pred_map = cv2.resize(predicted_map[i], new_size, interpolation=cv2.INTER_AREA)
                    score.append(roc_auc_score(ground_map.flatten(), re_pred_map.flatten()))

                all_scores += score
                if (len(all_scores) + 1) % 32 == 0:
                    print('Mean:', np.mean(all_scores), end='\r')

            configs['wild'] = np.mean(all_scores)

        all_configs.append(configs)

    df = pd.DataFrame.from_dict(all_configs)
    print(df)
    time = datetime.datetime.now().strftime('%H_%d_%m')
    if store:
        df.to_csv(wsl_summary_dir / f'wild_{time}')
