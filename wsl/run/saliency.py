import torch
import time
import numpy as np
from collections import defaultdict

import json
import cv2
from sklearn.metrics import average_precision_score as aupr

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from wsl.loaders.class_loaders import Loader
from wsl.locations import wsl_model_dir, wsl_plot_dir, known_tasks
from wsl.networks.medinet.backprop import BackProp
from wsl.networks.medinet.utils import box_to_map, rle2mask


def main(name: str, start: int, plot: bool):

    if name == 'all':
        model_dirs = wsl_model_dir.glob('rsna*')
    else:
        if 'rsna' in name:
            model_dirs = wsl_model_dir.glob(f'*{name}*')
        else:
            model_dirs = wsl_model_dir.glob(f'rsna*{name}*')

    model_dirs = list(model_dirs)
    num_model_dirs = 50
    print(f'Number of potential model directory matches = {len(model_dirs)}, but doing top {num_model_dirs} models for now.')
    model_dirs = model_dirs[start:start + num_model_dirs]

    if plot:
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))
        # change alpha values
        color_array[:, -1] = np.linspace(1.0, 0.0, ncolors)

    for m_idx, model_dir in enumerate(model_dirs):

        if 'debug' in str(model_dir):  # Debugging model
            print('Debugging model')
            continue

        elif not (model_dir / 'configs.json').exists():  # Model not completed
            print('Model not completed')
            continue

        else:
            with open(model_dir / 'configs.json') as f:
                configs = json.load(f)
            dataset = Loader(data=configs['data'],
                             split='test',
                             extension=configs['extension'],
                             classes=configs['classes'],
                             column=configs['column'],
                             variable_type=configs['variable_type'])
            print('Number of images -', len(dataset))

        print(f'Model {m_idx} : {model_dir}')

        if configs['data'] in known_tasks:
            task = known_tasks[configs['data']]

        checkpoint = torch.load(model_dir / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint['model'].gradient = None
        checkpoint['model'].eval()

        # currently hardcoded, should ideally be inferred from image
        org_size = (1024, 1024)
        new_size = (224, 224)

        all_scores = defaultdict(list)

        GD = BackProp(checkpoint['model'])
        GBP = BackProp(checkpoint['model'], True)

        start_time = time.time()

        for idx, data in enumerate(dataset):
            checkpoint['model'].zero_grad()
            name, img, label = data
            label = label.squeeze().cuda()

            if label != 1:
                continue

            # Make the ground map
            if task == 'detect':
                ground_map = box_to_map(dataset.df[dataset.df.Id == name].box.to_list(), np.zeros(org_size))
            elif task == 'segment':
                ground_map = np.zeros(org_size)
                eps = dataset.df[dataset.df.Id == name].EncodedPixels.to_list()
                for ep in eps:
                    ground_map += rle2mask(ep, np.zeros(org_size)).T

            ground_map = cv2.resize(ground_map, new_size, interpolation=cv2.INTER_NEAREST).clip(0, 1)

            # Make the saliency map
            checkpoint['model'].get_map = True
            if configs['wildcat']:
                _, _, wild, handle = checkpoint['model'](img.unsqueeze(dim=0).cuda().float())
                handle.remove()
                wild = torch.max(wild, dim=1)[0]
                wild = wild.squeeze().cpu().data.numpy()
                wild = (wild - wild.min()) / (wild.max() - wild.min())
                wild = cv2.resize(wild, new_size, interpolation=cv2.INTER_NEAREST)
            else:
                wild = np.zeros_like(ground_map)

            gcam = GD.generate_cam(img.unsqueeze(dim=0).cuda().float()).squeeze()

            checkpoint['model'].get_map = False
            grad = GD.generate_gradients(img.unsqueeze(dim=0).cuda().float())
            ig = GD.generate_integrated_gradients(img.unsqueeze(dim=0).cuda().float(), 25)

            sg = GD.generate_smooth_grad(img.unsqueeze(dim=0).cuda().float(), 5, 0.1, 0)
            sig = GD.generate_smooth_grad(img.unsqueeze(dim=0).cuda().float(), 5, 0.1, 10)

            gbp = GBP.generate_gradients(img.unsqueeze(dim=0).cuda().float())
            ggcam = np.multiply(gcam, gbp)

            all_scores['WILD'].append(aupr(ground_map.flatten(), wild.flatten()))
            all_scores['GRAD'].append(aupr(ground_map.flatten(), grad.flatten()))
            all_scores['SG'].append(aupr(ground_map.flatten(), sg.flatten()))
            all_scores['IG'].append(aupr(ground_map.flatten(), ig.flatten()))
            all_scores['SIG'].append(aupr(ground_map.flatten(), sig.flatten()))
            all_scores['GBP'].append(aupr(ground_map.flatten(), gbp.flatten()))
            all_scores['GCAM'].append(aupr(ground_map.flatten(), gcam.flatten()))
            all_scores['GGCAM'].append(aupr(ground_map.flatten(), ggcam.flatten()))

            if plot:
                row, col = 2, 5
                map_names = [['XRAY', 'WILD', 'GRAD', 'GCAM', 'GGCAM'], ['MASK', 'GBP', 'SG', 'IG', 'SIG']]
                maps = [[img, wild, grad, gcam, ggcam], [ground_map, gbp, sg, ig, sig]]
                x = LinearSegmentedColormap.from_list(name='rainbow', colors=color_array)
                plt.register_cmap(cmap=x)

                fig, ax = plt.subplots(row, col, figsize=(18, 8))
                for i in range(row):
                    for j in range(col):
                        ax[i, j].imshow(np.transpose(img, (1, 2, 0)))
                        if not (i == 0 and j == 0):
                            ax[i, j].imshow(maps[i][j], alpha=0.8, cmap='rainbow')
                        ax[i, j].text(0, 220, map_names[i][j], fontsize='x-large', color='white', weight='bold', bbox=dict(fill=True, linewidth=0))
                        ax[i, j].axis('off')
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                plt.savefig(f'{wsl_plot_dir}/saliency_{name}.png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

            print_str = f'{idx}: | '
            for key, value in all_scores.items():
                print_str += f'{key}-{int(np.mean(value) * 100)} | '
            print_str += str(round((time.time() - start_time) / (idx + 1), 2)) + ' s/img'
            print(print_str, end='\r')

        for key in all_scores.keys():
            configs[key] = np.mean(all_scores[key])
            print(key, ' ', configs[key])

        with open(model_dir / 'configs.json', 'w') as fp:
            json.dump(configs, fp)
