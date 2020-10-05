import torch
import numpy as np
from collections import defaultdict

import json
import cv2
from sklearn.metrics import average_precision_score as aupr

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from wsl.loaders.class_loaders import Loader
from wsl.locations import wsl_model_dir, wsl_plot_dir, known_tasks, known_layers
from wsl.networks.medinet.utils import box_to_map, rle2mask

from wsl.saliency.backprop import Gradient
from wsl.saliency.misc_functions import convert_to_grayscale
from wsl.saliency import gradcam

def main(name: str, start: int, plot: bool):
    
    if name == 'all':
        model_dirs = wsl_model_dir.glob('*')
    else:
        model_dirs = wsl_model_dir.glob(f'*{name}*')

    model_dirs = list(model_dirs)
    model_dirs = model_dirs[start:]
    
    print('Number of potential model directory matches =', len(model_dirs))
    
    if plot:
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))
        # change alpha values
        color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
    
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
                             regression=configs['regression'])
            print('Number of images -', len(dataset))

        print(f'Model {m_idx} : {model_dir}')
        
        if configs['data'] in known_tasks:
            task = known_tasks[configs['data']]
            
        checkpoint = torch.load(model_dir / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint['model'] = checkpoint['model'].module
        checkpoint['model'].get_map = True
        checkpoint['model'].eval()
        
        org_size = (1024, 1024)
        new_size = (224, 224)

        all_scores = defaultdict(list)
        
        GD = Gradient(checkpoint['model'], False)
        GBP = Gradient(checkpoint['model'], True)
        GCAM = gradcam.GradCam(checkpoint['model'], target_layer=known_layers[configs['network']])
        
        for idx, data in enumerate(dataset):
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
            if configs['wildcat']:
                with torch.set_grad_enabled(False):
                    wild = checkpoint['model'](img.unsqueeze(dim=0).cuda().float())
                    wild = wild.squeeze().cpu().data.numpy()
                wild = (wild - wild.min()) / (wild.max() - wild.min())
                wild = cv2.resize(wild, new_size, interpolation=cv2.INTER_NEAREST)
                
                all_scores['WILD'].append(aupr(ground_map.flatten(), wild.flatten()))
                
                if plot:
                    plt.figure(figsize=(4, 12))
                    x = LinearSegmentedColormap.from_list(name='rainbow', colors=color_array)
                    plt.register_cmap(cmap=x)
                    plt.subplot(1, 3, 1)
                    plt.imshow(np.transpose(img, (1, 2, 0)))
                    plt.subplot(1, 3, 2)
                    plt.imshow(ground_map, alpha = 0.8, cmap='rainbow')
                    plt.subplot(1, 3, 2)
                    plt.imshow(wild, alpha = 0.8, cmap='rainbow')
                    plt.savefig(f'{wsl_plot_dir}/wild_{name}.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()

            else:
                grad = GD.generate_gradients(img.unsqueeze(dim=0).cuda().float(), label)
                ig = GD.generate_integrated_gradients(img.unsqueeze(dim=0).cuda().float(), label, 100)
 
                sg = GD.generate_smooth_grad(img.unsqueeze(dim=0).cuda().float(), label, 5, 0.3, 0)
                sig= GD.generate_smooth_grad(img.unsqueeze(dim=0).cuda().float(), label, 5, 0.3, 100)
                
                gbp = GBP.generate_gradients(img.unsqueeze(dim=0).cuda().float(), label)

                print('gcam')
                gcam = GCAM.generate_cam(img.unsqueeze(dim=0).cuda().float(), label).squeeze()
                ggcam = np.multiply(gcam, gbp)
        
                all_scores['GRAD'].append(aupr(ground_map.flatten(), grad.flatten()))
                all_scores['SG'].append(aupr(ground_map.flatten(), sg.flatten()))
                all_scores['IG'].append(aupr(ground_map.flatten(), ig.flatten()))
                all_scores['SIG'].append(aupr(ground_map.flatten(), sig.flatten()))
                all_scores['GBP'].append(aupr(ground_map.flatten(), gbp.flatten()))
                all_scores['GCAM'].append(aupr(ground_map.flatten(), gcam.flatten()))
                all_scores['GGCAM'].append(aupr(ground_map.flatten(), ggcam.flatten()))
                
                if plot:
                    row, col = range(2), range(4)
                    map_names = [['MASK', 'GRAD', 'SG', 'IG'], ['SIG', 'GCAM', 'GBP', 'GGCAM']]
                    maps = [[ground_map, grad, sg, ig], [sig, gcam, gbp, ggcam]]
                    x = LinearSegmentedColormap.from_list(name='rainbow', colors=color_array)
                    plt.register_cmap(cmap=x)

                    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
                    for i in row:
                        for j in col:
                            ax[i, j].imshow(np.transpose(img, (1, 2, 0)))
                            ax[i, j].imshow(maps[i][j], alpha = 0.8, cmap='rainbow')
                            ax[i, j].text(0, 220, map_names[i][j], fontsize='x-large', color='white', weight='bold', bbox=dict(fill=True, linewidth=0))
                            ax[i, j].axis('off')
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    plt.savefig(f'{wsl_plot_dir}/saliency_{name}.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                
        for key in all_scores.keys():
            configs[key] = np.mean(all_scores[key])
            print(key, ' ', configs[key])
            
        with open(path / 'configs.json', 'w') as fp:
            json.dump(configs, fp)
