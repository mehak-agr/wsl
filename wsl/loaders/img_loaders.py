#!/usr/bin/python3
import sys
import random
import numpy as np
import pandas as pd
from ast import literal_eval
import pydicom
import skimage.io
from skimage import transform
from pathlib import Path
from typing import List, Dict, Optional

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
                 length: Optional[int] = None):

        self.datapath = wsl_data_dir / data
        self.data = data

        if data in known_extensions.keys():
            self.extension = known_extensions[data]
        else:
            self.extension = extension

        self.names = pd.read_csv(wsl_csv_dir / data / f'{split}.csv').Id.tolist()

        if length is not None:
            self.names = random.sample(self.names, min(len(self.names), length))

        self.new_size = (224, 224)
        self.image_transforms = Compose([
            Resize(self.new_size),
            RepeatChannel(repeats=3),
            CastToType(dtype=np.float32),
            ToTensor()])

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

        return name, img

    def __len__(self):
        return len(self.names)
