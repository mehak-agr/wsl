import json
import time
import random
import numpy as np
from umap.umap_ import UMAP
from typing import Any, List
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from wsl.locations import wsl_model_dir, known_extensions
from wsl.loaders.img_loaders import Loader
import torch
from torch.utils.data import DataLoader

colour_list = ['lightsalmon', 'orangered', 'indianred', 'brown', 'palegreen', 'darkseagreen',
               'greenyellow', 'darkolivegreen', 'lightskyblue', 'deepskyblue', 'cyan', 'dodgerblue']


def plot(features, labels, classes, path):
    print(features.shape, labels.shape)
    
    features, labels = shuffle(features, labels)
    
    print('Plotting UMAP...', end='')
    features = features.reshape(features.shape[0], -1)
    embedding = UMAP(n_neighbors=20, min_dist=1, metric='correlation', random_state=1, transform_seed=1).fit_transform(features)
    colours = ListedColormap(colour_list[:len(classes)])
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=3, alpha=1, cmap='plasma')

    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='best',ncol=1, fontsize=6)
    plt.savefig(str(path / 'umap.png'), dpi=300)
    plt.close()
    print('done.')


def main(debug: bool = True, model: str = 'rsna_pneumonia_lr0.0001_bs32_adam_densenet121_wildcat_maps1_alpha0.05_flaneur',
         datasets: Any = ['cancer_mgh', 'cancer_dmist2', 'cancer_dmist3', 'cancer_dmist4']):

    path = wsl_model_dir / model
    print(f'Model: {path}')
    assert path.exists()

    if (path / 'configs.json').exists():  # Model not completed
        with open(path / 'configs.json') as f:
            configs = json.load(f)
            # print(configs)
    else:
        print('Incomplete model')
        return
    
    checkpoint = torch.load(path / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint['model'].eval()

    features = {}
    print(checkpoint['model'].module._modules.keys())

    if configs['wildcat']:
        layer_name = 'classifier'
    else:
        layer_name = 'pool'

    def hook(layer, inp, out):
        if layer_name not in features:
            features[layer_name] = out.detach().data.view(out.size(0), -1)
        else:
            features[layer_name] = torch.cat((features[layer_name], out.detach().data.view(out.size(0), -1)), dim=0)
    handles = checkpoint['model'].module._modules[layer_name].register_forward_hook(hook)

    dataset_classes = []
    for dataset_id, dataset in enumerate(datasets):
        loader = Loader(data=dataset,
                        split='valid',
                        extension=known_extensions[dataset],
                        length=500)
        dataloader = DataLoader(  # type: ignore
            loader, batch_size=configs['batchsize'], num_workers=4,
            pin_memory=True, shuffle=True)
        print(f'Length of {dataset}: {len(loader.names)}')
        dataset_classes += [dataset_id] * len(loader)

        start = time.time()
        with torch.set_grad_enabled(False):
            for idx, data in enumerate(dataloader):
                imgs = data[1].cuda().float()
                _ = checkpoint['model'](imgs)
                speed = configs['batchsize'] * idx // (time.time() - start)
                print('Iter:', idx, 'Speed:', int(speed), 'img/s', end='\r', flush=True)
        print('Total time:', time.time() - start, 'secs')
    
    plot(features[layer_name].cpu().detach().numpy(), np.asarray(dataset_classes), datasets, path)


if __name__ == '__main__':
    main()
