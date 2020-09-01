#!/usr/bin/python3
import numpy as np
import pandas as pd
from functools import reduce
import pydicom
from pathlib import Path

from monai.data import Dataset
from monai.transforms import (
    Compose,
    RepeatChannel,
    Resize,
    CastToType,
    ToTensor
)

from wsl.locations import wsl_data_dir


class BinaryLoader(Dataset):
    def __init__(self, data: str, split: str, extension: str, debug: bool):

        self.datapath = wsl_data_dir / data / 'images'
        self.extension = extension

        df = pd.read_csv(wsl_data_dir / data / 'info.csv')
        df = df.drop_duplicates(subset='Id', keep='first', ignore_index=True)
        self.names = pd.read_csv(wsl_data_dir / data / f'{split}.csv').Id.tolist()
        self.labels = reduce(pd.DataFrame.append, map(lambda i: df[df.Id == i], self.names)).Target.tolist()
        if debug:
            self.names = self.names[0:100]
            self.labels = self.labels[0:100]

        self.pos_weight = [round((len(self.labels) - sum(self.labels)) / sum(self.labels), 2)]

        self.common_transforms = Compose([
            Resize((224, 224)),
            RepeatChannel(repeats=3),
            CastToType(dtype=np.float32),
            # NormalizeIntensity(subtrahend=np.array([0.485, 0.456, 0.406]), divisor=np.array([0.229, 0.224, 0.225])),
            ToTensor()]
        )

    def load_image(self, path: Path):
        ref = pydicom.dcmread(path)
        img = ref.pixel_array
        pi = ref.PhotometricInterpretation
        if pi.strip() == 'MONOCHROME1':
            img = -img
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = self.datapath / f'{name}.{self.extension}'
        img = self.load_image(path)
        return self.common_transforms(img), self.labels[idx]

    def __len__(self):
        return len(self.names)
