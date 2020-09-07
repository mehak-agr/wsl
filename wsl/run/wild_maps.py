#!/usr/bin/python3
import torch
import json
import cv2
from wsl.locations import wsl_model_dir
from wsl.loaders.loaders import Loader
from wsl.network.utils import box_to_map


def main():
    models = wsl_model_dir.glob('*')
    print(models)
    for idx, path in enumerate(models):
        print(f'Model {idx} : {path.stem}')

        if 'debug' in path.stem:
            print('Debugging model: Skipping')
            continue
        if (path / 'config.json').exists():
            with open(path / 'configs.json') as f:
                configs = json.load(f)
            print(configs)
        else:
            print('Model not completed, put it on resume in train: Skipping')
            continue

        checkpoint = torch.load(path / 'best.pt', map_location='cuda:0' if torch.cuda_is_available() else 'cpu')
        checkpoint['model'].get_map = True
        dataset = Loader(data=configs['data'],
                         split='valid',
                         extension=configs['extension'],
                         classes=configs['classes'],
                         col_name=configs['column'],
                         regression=configs['regression'])

        with torch.set_grad_enabled(False):
            for idx, data in enumerate(dataset):
                img, label = data
                name = dataset.names[idx]
                assert (dataset.labels[idx] == label)

                ground_map = box_to_map(dataset.df[dataset.df == name].to_dict(orient='split')['data'][:][1:],
                                        org_size=(1024, 1024), new_size=(224, 224))

                predicted_map = checkpoint['model'](img.unsqueeze(dim=0).cuda().float()).cpu().data
                predicted_map = cv2.resize(predicted_map, img.shape, interpolation=cv2.NEAREST_AREA)

                # To be implemented different metric calculations
                # To be implemented making of overall results df


main()


