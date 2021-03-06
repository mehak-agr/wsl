#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
from ast import literal_eval
import pydicom
import skimage.io
from skimage import transform
from pathlib import Path
from typing import List, Dict

from monai.data import Dataset
from monai.transforms import (
    Compose,
    RepeatChannel,
    Resize,
    CastToType,
    ToTensor,
    Affine
)
import torch
from wsl.locations import wsl_data_dir, wsl_csv_dir, known_extensions


class Loader(Dataset):
    def __init__(self, data: str, split: str, extension: str,
                 classes: int, column: str, variable_type: str,
                 augmentation: bool = False, debug: bool = False):

        if classes != 1:
            print('Note: Ensure all labels are of a single type.')

        self.datapath = wsl_data_dir / data
        self.data = data
        self.classes = classes

        if data in known_extensions.keys():
            self.extension = known_extensions[data]
        else:
            self.extension = extension

        df = pd.read_csv(wsl_csv_dir / data / 'info.csv', converters={column: literal_eval, 'box': literal_eval})
        self.df = df
        df = df.drop_duplicates(subset='Id', keep='first', ignore_index=True)
        Ids = pd.read_csv(wsl_csv_dir / data / f'{split}.csv').Id.tolist()
        df = df[df.Id.isin(Ids)]

        self.names = df.Id.to_list()
        self.labels = df[column].tolist()
        self.variable_type = variable_type

        if debug:
            self.names = self.names[0:100]
            self.labels = self.labels[0:100]

        self.new_size = (224, 224)
        self.image_transforms = Compose([
            Resize(self.new_size),
            RepeatChannel(repeats=3),
            CastToType(dtype=np.float32),
            ToTensor()])

        self.augmentation = augmentation
        if augmentation:
            self.augmentation = Affine(
                rotate_params = np.pi / 6,
                scale_params = (1.2, 1.2),
                translate_params = (50, 50),
                padding_mode = 'zeros'
            )
        else:
            self.augmentation = None

        if self.variable_type != 'categorical':
            if classes == 1:
                self.labels = [[x] for x in self.labels]
            else:
                self.class_names = self.labels[0].keys()
                print('\nClass List: ', self.class_names)
                self.labels = [list(x.values()) for x in self.labels]

            # only matters for balanced case for binary variable type
            self.pos_weight = [round((len(col) - sum(col)) / sum(col), 2) for col in zip(*self.labels)]

    def load_image(self, path: Path):
        if self.extension == 'dcm' or self.extension == '':
            ref = pydicom.dcmread(path)
            img = ref.pixel_array
            pi = ref.PhotometricInterpretation
            if pi.strip() == 'MONOCHROME1':
                img = -img
        elif self.extension == 'npy':
            img = np.load(path)
            img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        else:
            img = skimage.io.imread(path, as_gray=True)

        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def __getitem__(self, idx):
        name = self.names[idx]
        if self.extension != '':
            path = self.datapath / f'{name}.{self.extension}'
        else:
            path = self.datapath / name

        img = self.load_image(path)
        img = self.image_transforms(img)
        if self.augmentation is not None:
            img = self.augmentation(img)

        label = self.labels[idx]
        if self.variable_type == 'categorical':
            return name, img, label
        else:
            return name, img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.names)
