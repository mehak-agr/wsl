# PyTorch modules
import torch
from torch import nn 
from torch.utils import data 
import torch.nn.functional as F 
from torchvision import transforms, models

# other modules
import os
import pandas as pd
import numpy as np
from PIL import Image
import random
import pydicom
import cv2


class MGH_Dataset(data.Dataset):
    """ 
    Create dataset representation of MGH data

    """  
    def __init__(self, patient_table, epoch_size, transform=None):
        """
        Args:
            patient_table (pd.dataframe): dataframe containing relative image paths, abnormal_lung, and other metadata
            image_dir (string): directory containing image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.patient_table = patient_table
        self.transform = transform
        self.epoch_size = epoch_size 
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
 
    def __len__(self):
        return self.epoch_size
 
    def __getitem__(self, idx): 
        
        table_A = self.patient_table

        # if no change 
        pick0 = random.choice(range(len(table_A)))

        img0 = Image.open(table_A.iloc[pick0]['Path'])

        label = float(table_A.iloc[pick0]['mRALE'])

        path = self.patient_table.iloc[pick0]['Path']

        if self.transform is not None:
            img0 = self.transform(img0)

        return img0, label, path
