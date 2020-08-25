import requests
import datetime
from pathlib import Path

from locations import wsl_model_dir

from torch import nn, optim

from wsl.networks.engine import Engine
from wsl.networks.architecture import Architecture

from wsl.loaders import rsna, chex, mura, custom

loader_map = {'rsna': rsna.Loader(),
              'chex': chex.Loader(),
              'mura': chex.Loader(),
              'custom': custom.Loader()}


def main(data: str,
         dicom: bool,
         classes: int,
         network: str,
         depth: int,
         wildcat: bool,
         pretrained: bool,
         optim: str,
         resume: bool,
         name: str = '',
         lr: float,
         batchsize: int,
         workers: int,
         patience: int,
         balanced: int,
         maps: int,
         alpha: float,
         k: int):

    # ------------------------------------------------------
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
        print('Model Name:', mname)

        if wildcat:
            type_str = f'_wildcat_maps{maps}_alpha{alpha}_k{k}'
            model = architecture.Wildcat(network, depth, classes, maps, alpha, k, pretrained)
        else:
            model = architecture.Base(network, depth, classes, pretrained)

        full_mname = (f'{data}{'_dicom' if dicom else ''}_' +
                      f'lr{lr}_bs{batchsize}_{optimizer}{'_pre' if pretrained else ''}{'_bal' if balanced else ''}_' +
                      f'{network}{depth}{type_str if 'wildcat' else ''}_{mname}')
        model_dir = wsl_model_dir / full_mname

    # ------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    model = Architecture(network, depth, wildcat, classes, maps, alpha, k, pretrained)

    if optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.1, 1e-4)
    elif optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f'{optim} is not supported')

    # ------------------------------------------------------
