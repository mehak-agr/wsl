import torch
import torch.nn as nn
# from unet import UNet
import segmentation_models_pytorch as smp

# model = smp.Unet()
# from dice import dice_coeff, SoftDiceLoss, BinaryDiceLoss
from dice import dice_coeff, MixedLoss
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import imageio
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split

import random
import argparse
import logging
import os
import sys
from tqdm import tqdm

# Class to create data set (image names + labels)
class ChestXRay(Dataset):
    def __init__(self, image_list_file, label_file, datadir, transform=None, norm_transform=None): # image_list_file refers to the csv file with images names + labels
        
        # Attributes 
        image_names = []
        masks = []
        labels = []
        for i, (ind, row) in enumerate(image_list_file.iterrows()):
            # if label_file['label'][i] == 1:
            image_name = row['name'] # Tweak to use appropriate file directory            
            image_names.append(datadir + "train/" + image_name)
            mask = row['name']
            masks.append(datadir + "mask/" + mask)  
        for i, (ind, row) in enumerate(label_file.iterrows()):
            labels.append(row['label'])

        print("Same Lengths: {}".format(len(image_names) == len(labels)))
        self.image_names = image_names
        self.masks = masks
        self.labels = labels
        self.transform = transform
        self.norm_transform = norm_transform

    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = imageio.imread(image_name)
        #make a [3,320,320] array to match with RGB input requirement
        image = Image.fromarray(image)
        image = image.convert('RGB')

        mask_name = self.masks[index]
        label_name = self.labels[index]
        mask = imageio.imread(mask_name)

        mask = Image.fromarray(mask)
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.norm_transform is not None:
            image = self.norm_transform(image)

        if (torch.max(mask).item() > 1):
            mask /= 255
        mask[mask >= 0.5] = 1
        mask[mask < .5] = 0
        return image, mask
    
    def __len__(self):
        return len(self.image_names)

# samples classes in 1:1 ratio
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max (which is positive class in this case)
        for label in [1]:
            while len(self.dataset[label]) < (1 * self.balanced_max):
                self.dataset[label].append(random.choice(self.dataset[label]))

        print("Negative cases {} and positive cases {}".format(len(self.dataset[0]), len(self.dataset[1])))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < len(self.dataset[self.keys[self.currentkey]]) - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if dataset.labels is not None:
            return dataset.labels[idx]
        else:
            # Trying guessing
            print("guessing..")
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

parser = argparse.ArgumentParser(description='Segmentation Training')
parser.add_argument('--p',
                    help='percents', type=float, default=1)
parser.add_argument('--weight',
                    help='weight', type=float, default=1)
def run_arg():
    args = parser.parse_args()

    percents = [args.p]
    print("working on these percents: {}".format(percents))
    epochs = 75
    dataDir = '../PNGs/'
    trBatchSize = 1
    # label_path = './dataCreationCSVs/'

    # for nishanth
    label_path = '../../Pneumothorax_Data/csvs/edited_train_rle.csv'
    

    # load the data
    for percent in percents:
        # print(str(percent))

        # df = pd.read_csv(label_path + "Supervised_" + str(float(percent*100)) + "%.csv")
        # df = df.drop_duplicates(subset=['name'])
        # initial_length = len(df)
        # # df = df[df['label']==1]
        # # print(f"Length ratio: {len(df)/initial_length}")
        # x_train = df[['name']]
        # y_train = df[['label']]
        # print("Length of Train", len(x_train))

        # # make valid labels
        # df_val = pd.read_csv(label_path + "Val_" + str(float(percent * 100)) + "%.csv")
        # df_val = df_val.drop_duplicates(subset=['name'])
        # # df_val = df_val[df_val['label']==1]
        # x_valid = df_val[['name']]
        # y_valid = df_val[['label']]

        # # make test labels 
        # df_test = pd.read_csv(label_path + "test.csv")
        # df_test = df_test.drop_duplicates(subset=['name'])
        # # df_test = df_test[df_test['label']==1]
        # x_test = df_test[['name']]
        # y_test = df_test[['label']]


        # for nishanth
        print("for nishanth")
        df = pd.read_csv(label_path)
        x = df[['name']]
        y = df[['label']]
        xt, x_test, yt, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
        x_train, x_valid, y_train, y_valid = train_test_split(xt, yt, test_size=0.1, random_state=1)

        print("lengths!..", len(x_train), len(x_valid), len(x_test))

        # TRANSFORM DATA
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Create list of transformations
        transformList = []
        transformList.append(transforms.Resize((320, 320)))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformSequence_train = transforms.Compose(transformList) # Compose all these transformations (later apply to data set)
        transform_normalize = transforms.Compose([normalize])

        # Same thing but for validation and testdata
        transformList = []
        transformList.append(transforms.Resize((320, 320)))
        transformList.append(transforms.ToTensor())
        transformSequence_valid = transforms.Compose(transformList)


        # LOAD DATASET
        print("loading train...")
        datasetTrain = ChestXRay(x_train, y_train, dataDir, transformSequence_train, transform_normalize)
        print("loading valid...")
        datasetValid = ChestXRay(x_valid, y_valid, dataDir, transformSequence_valid, transform_normalize)

        print("dataloading..", 'train')
        dataLoaderTrain = DataLoader(dataset=datasetTrain, sampler=BalancedBatchSampler(datasetTrain), batch_size=4*trBatchSize, num_workers=4, pin_memory=True)
        print('dataloading... valid')
        dataLoaderValid = DataLoader(dataset=datasetValid, batch_size=4*trBatchSize, shuffle=False, num_workers=4, pin_memory=True)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = smp.Unet(activation=None)

        model = torch.nn.DataParallel(model).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

        print("using device {}".format(device))

        # run epochs
        accumulate = 2
        patience = 15
        min_loss = 100000
        criterion = MixedLoss(args.weight,2.)

        model2 = model
        for epoch in range(epochs):
            # train the model
            model.train()
            epoch_loss = []
            with tqdm(total=len(dataLoaderTrain), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                # train model
                for img, mask in dataLoaderTrain:
                    optimizer.zero_grad()
                    img = img.cuda()
                    mask = mask.cuda()
                    mask = mask.float()
                    # make sure to use grad_enabled
                    with torch.set_grad_enabled(True):
                        pred = model(Variable(img)) # [N, 2, H, W]
                        pred = pred.float()
                        loss = criterion(pred, Variable(mask)) 
                        epoch_loss.append(loss.item())
                        # lets accululate loss and only step optimizer every 2 steps
                        loss /= accumulate
                        loss.backward()
                        if (epoch + 1) % accumulate == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    pbar.set_postfix(**{'loss (batch)': np.mean(epoch_loss)})
                    pbar.update(1)

            # Do Validation at the end of each epoch
            with torch.set_grad_enabled(False):
                model.eval()
                tot = []
                dices = []
                # go through validation files
                for img2, mask2 in tqdm(dataLoaderValid):
                    img2 = img2.cuda()
                    true_mask = mask2.cuda()
                    pred = model(Variable(img2))
                    tot.append(criterion(pred, Variable(true_mask)).item())
                    pred = (nn.Sigmoid()(pred) > 0.5).float()
                    dice = dice_coeff(pred, Variable(true_mask.float())).item()
                    dices.append(dice)

                val_loss = np.mean(tot)
                print("Dice Coeff...{}\nVal Loss...{}".format(np.mean(dices), val_loss))

                if val_loss < min_loss:
                    min_loss = val_loss
                    patience = 15
                    print("saving model")
                    torch.save(model.state_dict(), './unet_models/nishanthr_unet_weight{}_sig{}.pth.tar'.format(args.weight, float(percent *100)))

                else:
                    patience -= 1
                if patience == 0:
                    break
            
            scheduler.step(val_loss)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_arg()
