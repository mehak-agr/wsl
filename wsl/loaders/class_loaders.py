#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
from ast import literal_eval
import pydicom
import skimage.io
from pathlib import Path

from monai.data import Dataset
from monai.transforms import (
    Compose,
    RepeatChannel,
    Resize,
    CastToType,
    ToTensor
)
import torch
from wsl.locations import wsl_data_dir, wsl_csv_dir, known_extensions


class Loader(Dataset):
    def __init__(self, data: str, split: str, extension: str,
                 classes: int, column: str,
                 regression: bool, debug: bool = False):

        if regression and classes != 1:
            print('Support for multi-class regression is not available.')
            sys.exit(1)

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

        if debug:
            self.names = self.names[0:100]
            self.labels = self.labels[0:100]

        self.image_transforms = Compose([
            Resize((224, 224)),
            RepeatChannel(repeats=3),
            CastToType(dtype=np.float32),
            ToTensor()])

        if regression:
            self.lmax = df[column].max()
            self.lmin = df[column].min()
            self.labels = [[round((x - self.lmin) / self.lmax, 2)] for x in self.labels]
        else:
            if classes == 1:
                self.labels = [[x] for x in self.labels]
            else:
                self.class_names = self.labels[0].keys()
                self.labels = [list(x.values()) for x in self.labels]

            self.pos_weight = [round((len(col) - sum(col)) / sum(col), 2) for col in zip(*self.labels)]

    def load_image(self, path: Path):
        if self.extension == 'dcm' or self.extension == '':
            ref = pydicom.dcmread(path)
            img = ref.pixel_array
            pi = ref.PhotometricInterpretation
            if pi.strip() == 'MONOCHROME1':
                img = -img
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

        label = self.labels[idx]
        label = torch.Tensor(label)

        return name, img, label

    def __len__(self):
        return len(self.names)
