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
                 classes: int, column: str, debug: bool = False):
        self.datapath = wsl_data_dir / data
        self.data = data
        self.classes = classes
        self.column = column

        if data in known_extensions.keys():
            self.extension = known_extensions[data]
        else:
            self.extension = extension

        df = pd.read_csv(wsl_csv_dir / data / 'info.csv', converters={column: literal_eval, 'box': literal_eval})
        self.df = df
        Ids = pd.read_csv(wsl_csv_dir / data / f'{split}.csv').Id.tolist()
        df = df[df.Id.isin(Ids)]
        self.max_boxes = df['Id'].value_counts().max()
        self.names = list(set(df.Id.to_list()))
        if debug:
            self.names = self.names[0:100]

        self.image_transforms = Compose([
            # Resize((224, 224)),
            RepeatChannel(repeats=3),
            CastToType(dtype=np.float32),
            ToTensor()])
        
    def load_image(self, path: Path):
        if self.extension == 'dcm':
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
    
    def load_boxes(self, name: str, size: float):
        labels = self.df[self.df.Id == name][self.column].to_list()
        
        if sum(labels) == 0:
            label = 0
            boxes = -1 * torch.ones((self.max_boxes, 5))
        else:
            label = 1
            boxes = torch.Tensor(self.df[self.df.Id == name].box.to_list())
            # boxes = boxes * 224 / size
            labels = torch.zeros((len(boxes), 1))  # Box Label = 0 for positive
        
            boxes = torch.cat((boxes, labels), dim=1)
            filler = [[-1, -1, -1, -1, -1]] * (self.max_boxes - len(boxes))
            boxes = torch.cat((boxes, torch.Tensor(filler)))

        return boxes, label
    
    def __getitem__(self, idx):
        name = self.names[idx]
        path = self.datapath / f'{name}.{self.extension}'

        img = self.load_image(path)
        size = int((img.shape[0] + img.shape[1]) / 2)
        img = self.image_transforms(img)
        
        boxes, label = self.load_boxes(name, size)
        return img, boxes, label, name

    def __len__(self):
        return len(self.names)
