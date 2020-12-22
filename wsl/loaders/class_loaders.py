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
    ToTensor
)
import torch
from wsl.locations import wsl_data_dir, wsl_csv_dir, known_extensions


class Loader(Dataset):
    def __init__(self, data: str, split: str, extension: str,
                 classes: int, column: str, regression: bool,
                 augmentation: bool = False, debug: bool = False):

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

        self.augmentation = augmentation
        self.rand_rotation: float = 20
        self.rand_scale: List = [0.8, 1.2]
        self.rand_shear: float = 0.25
        self.rand_translation: List = [50.0, 50.0]
        self.rand_roll: Dict = {'h': False, 'w': False}
        self.rand_flip: List = [False, True, False]
        self.noise_factor: float = 0.05
        self.noise_dist = np.random.normal  # or np.random.poisson

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

    def augment(self, img: np.array):
        if self.rand_rotation or self.rand_scale or self.rand_shear or self.rand_translation:
            # Perform a random affine transformation
            # Rotation
            theta = np.random.uniform(-self.rand_rotation * np.pi / 180.0, self.rand_rotation * np.pi / 180.0) if self.rand_rotation else 0.0
            # Shear
            shear_factor = np.random.uniform(-self.rand_shear, self.rand_shear) if self.rand_shear else 0.0
            # Anisotropic scaling in two dimensions
            scale = np.random.uniform(self.rand_scale[0], self.rand_scale[1], size=2) if self.rand_scale else (1.0, 1.0)
            # Translations
            if self.rand_translation:
                translation = (
                    np.random.uniform(-self.rand_translation[0], self.rand_translation[0]),
                    np.random.uniform(-self.rand_translation[1], self.rand_translation[1])
                )
            else:
                translation = (0.0, 0.0)

            # Construct a transform that moves the center to the origin, performs the affine transform, and then moves
            # back
            affine_xform = transform.AffineTransform(
                rotation=theta, shear=shear_factor, scale=scale, translation=translation
            )
            center_xform = transform.AffineTransform(translation=(-img.shape[1] / 2, -img.shape[2] / 2))
            inv_center_xform = transform.AffineTransform(translation=(img.shape[1] / 2, img.shape[2] / 2))
            full_xform = center_xform + affine_xform + inv_center_xform

            img = transform.warp(np.moveaxis(img, 0, -1), full_xform, cval=-1000, order=1, preserve_range=True)
            img = np.moveaxis(img, -1, 0)

        if self.rand_roll['h']:
            h_roll = round(self.rand_roll['h'] * np.random.random())
            img = np.roll(img, h_roll, axis=1)

        if self.rand_roll['w']:
            w_roll = round(self.rand_roll['w'] * np.random.random())
            img = np.roll(img, w_roll, axis=2)

        if sum(self.rand_flip):
            for i in range(img.ndim):
                if np.random.randint(0, self.rand_flip[i] + 1):
                    img = np.flip(img, axis=i)

        if self.noise_factor:
            noise = self.noise_factor * self.noise_dist(size=img.shape)
            img += noise

        return img

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
        if self.augmentation:
            img = self.augment(img)
        img = self.image_transforms(img)

        label = self.labels[idx]
        label = torch.Tensor(label)

        return name, img, label

    def __len__(self):
        return len(self.names)
