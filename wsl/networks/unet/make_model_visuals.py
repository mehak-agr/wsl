# makes visualizations and goes deeper into the accuracies of the models

import pandas as pd 
import numpy as np 
from unet import UNet
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sn
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
from dice import dice_coeff
from sklearn.model_selection import train_test_split
import cv2
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
# create argument parsers (takes in percent and weight)
parser = argparse.ArgumentParser(description='Check segmenation performance')
parser.add_argument('--p',
                    help='percent of data')
parser.add_argument('--w', 
                    help='weight')

# Class to create data set (image names + labels)
class ChestXRay(Dataset):
    def __init__(self, image_list_file, label_file, datadir, transform=None, norm_transform=None): # image_list_file refers to the csv file with images names + labels
        
        # Attributes 
        image_names = []
        masks = []
        labels = []
        for i, (ind, row) in enumerate(image_list_file.iterrows()):
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

        # fix the mask to be 0,1 values
        if (torch.max(mask).item() > 1):
            mask /= 255
        mask[mask >= 0.5] = 1
        mask[mask < .5] = 0

        return image, mask, image_name, label_name
    
    def __len__(self):
        return len(self.image_names)

# load the model that we want to use for segmentation
def load_model(weight, percent):
    print('loading model...', sep=' ')
    model = smp.Unet()
    state_dict = torch.load('./unet_models/nishanthr_unet_weight{}_sig{}.pth.tar'.format(float(weight), float(percent)))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    print('model loaded.')
    return model

# load the data for purpose that we are interested in
# primarily used for test, returns a dataloader
def load_data(data_name):
    # if data_name == 'train':
    #     print('using train...')
    #     df = pd.read_csv('./dataCreationCSVs/Supervised_100.0%.csv')

    # elif data_name == 'valid':
    #     print('using validate...')
    #     df = pd.read_csv('./dataCreationCSVs/Val_100.0%.csv')

    # else:
    #     print('using test...')
    #     df = pd.read_csv('./dataCreationCSVs/test.csv')

    print("for nishanth")
    label_path = '../../Pneumothorax_Data/csvs/edited_train_rle.csv'

    df = pd.read_csv(label_path)
    x = df[['name']]
    y = df[['label']]
    xt, x_test, yt, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    dataDir = '../PNGs/'
    trBatchSize = 1

    # make transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.Resize((320, 320)))
    transformList.append(transforms.ToTensor())
    transformSequence = transforms.Compose(transformList)
    transform_normalize = transforms.Compose([normalize])

    # make dataloader
    # datasetLoad = ChestXRay(df[['name']], df[['label']], dataDir, transformSequence, transform_normalize)
    datasetLoad = ChestXRay(x_test, y_test, dataDir, transformSequence, transform_normalize)
    dataLoaderImg = DataLoader(dataset=datasetLoad, batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=True)

    return dataLoaderImg

# method to run the full test
def run_test():
    args = parser.parse_args()
    # load model and dataload
    model = load_model(weight=int(args.w), percent=100*float(args.p))
    model = model.cuda()
    dataLoad = load_data('test')

    # create arrays to store data
    model_classify = [[0,0],[0,0]]
    model_guess_size = [[0,0],[0,0]]
    model_actual_size = [[0,0],[0,0]]
    dice, dice_ones = [], []
    saved = False
    model.eval()
    true_labels = []
    guesses = []
    names = []
    save_image = './im_results_nish/'
    with tqdm(total=len(dataLoad), desc=f'Testing...', unit='img') as pbar2:
        # iterate through the dataLoader
        count = 1
        sizes = set()
        with torch.set_grad_enabled(False):
            for X2, y2, im_name, label in dataLoad:
                # get the output of our model, as well as its dice coeff
                X2 = X2.cuda()
                true_mask = y2.cuda()
                pred = torch.sigmoid(model(X2)).cuda()
                true_labels.append(label.item())
                pr = pred.cpu().detach().numpy()
                if len(np.unique(pr)) == 1:
                    guesses.append(np.unique(pr)[0])
                else:
#                    s = np.true_divide(pr.sum(),(pr!=0).sum())
                    # s = np.median(pr[pr != 0])
                    s = stats.mode(pr[pr != 0], axis=None)[0][0]
                    print(s)
                    guesses.append(s) 
                # guesses.append(np.amax(pred.cpu().detach().numpy()))
                names.append(im_name[0].split("/")[-1])
                # update the tdqm bar
                pbar2.update()
                mask = pred.cpu().detach().numpy() * 255
                # mask = mask.transpose((3,2,1,0))
                mask[mask > 255] = 255
                mask[mask < 0] = 0
                mask = mask.reshape(320,320)
                #print(np.unique(mask))
                image = Image.fromarray(mask.astype(np.uint8))

                im_name = im_name[0].split("/")[-1]
                image.save(save_image + im_name, 'PNG')

                # cast values to necessary types
                pred = (pred > 0.5).float()
                d = dice_coeff(pred, true_mask).item()
                dice.append(d)

                # put predictions and masks to cpu to do computations
                prediction = pred.cpu().detach().numpy()
                true_mask = true_mask.cpu()
                
                actual_mask = true_mask.numpy()
                label = int(label)

                # gets the number of nonzero elements (mask)
                pred_size = np.count_nonzero(prediction)
                actual_size = np.count_nonzero(actual_mask)

                # Make sure if our label is 0, our mask size is also 0
                if label == 0:
                    assert actual_size == label
                else:
                    dice_ones.append(d)

                # fill in the table values as necessary
                guess = 0 if pred_size < 1 else 1
                model_classify[label][guess] += 1

                # update matrices as necessary to include model outputs
                model_guess_size[label][guess] = (model_guess_size[label][guess] * (count-1) + pred_size) / count
                model_actual_size[label][label] = (model_actual_size[label][label] * (count-1) + actual_size) / count
                count += 1

    df = pd.DataFrame({"names":names, "true": true_labels, "predicted": guesses})
    df.to_csv("./im_results_nish/outputs.csv", index=False)
    print("AUC_ROC: {}\nAUPRC: {}".format(roc_auc_score(true_labels, guesses), average_precision_score(true_labels, guesses))) 
    # prints the arrays that we are interested in
    names = ['classification', 'guess_size', 'actual_size']
    model_grids = [model_classify, model_guess_size, model_actual_size]
    print("drawing grids...")
    for index in range(len(model_grids)):
        arr = model_grids[index]
        print(arr)
    
    # print the final scores for both overall dice and dice score for positive images
    print("dice score {}\ndice ones score {}".format(np.mean(dice), np.mean(dice_ones)))
    print("done!")

if __name__ == '__main__':
    run_test()
