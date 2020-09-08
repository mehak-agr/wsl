#!/usr/bin/python3
import torch
import json
import cv2
from sklearn.metrics import roc_auc_score
from wsl.locations import wsl_model_dir
from wsl.loaders.loaders import Loader
from wsl.networks.utils import box_to_map


def main():
    models = wsl_model_dir.glob('*')
    for idx, path in enumerate(models):
        print(f'Model {idx} : {path}')

        if 'debug' in str(path):
            print('Debugging model: Skipping')
            continue
        if (path / 'configs.json').exists():
            with open(path / 'configs.json') as f:
                configs = json.load(f)
            # print(configs)
        else:
            print('Model not completed, put it on resume in train: Skipping')
            continue

        checkpoint = torch.load(path / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint['model'].get_map = True
        dataset = Loader(data=configs['data'],
                         split='valid',
                         extension=configs['extension'],
                         classes=configs['classes'],
                         col_name=configs['column'],
                         regression=configs['regression'])
        
        org_size = (1024, 1024)
        new_size = (224, 224)
        sigmoid = torch.nn.Sigmoid().cuda()

        with torch.set_grad_enabled(False):
            for idx, data in enumerate(dataset):
                img, label = data
                name = dataset.names[idx]
                labels = dataset.labels[idx]

                predicted_map = checkpoint['model'](img.unsqueeze(dim=0).cuda().float()).squeeze(dim=0)
                print(predicted_map.shape, checkpoint['model'].get_map)
                predicted_map = sigmoid(predicted_map).cpu().data.numpy()
                
                for i, label in enumerate(labels):
                    if label == 0:
                        continue

                    ground_map = box_to_map(dataset.df[dataset.df.Id == name].to_dict(orient='row'),
                                            configs['column'], org_size, new_size)
                    re_pred_map = cv2.resize(predicted_map[i], new_size, interpolation=cv2.INTER_AREA)

                    print(ground_map.mean(), re_pred_map.mean())
                    score = roc_auc_score(ground_map.flatten(), re_pred_map.flatten())
                    print(score)

                # To be implemented different metric calculations
                # To be implemented making of overall results df


main()


