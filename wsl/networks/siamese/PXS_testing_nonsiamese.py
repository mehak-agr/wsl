# -*- coding: utf-8 -*-
'''
PXS3 non siamese, MSE loss only
'''

# PyTorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.autograd import Variable

# other modules 
import os
from datetime import datetime
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics
import scipy
import pickle
from tqdm import tqdm
from PIL import Image
import seaborn as sns

import cv2
import pydicom
from medpy.io.load import load
 
 # WORKING DIRECTORY
working_path = '/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/'
os.chdir(working_path)

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='4' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# loading model 
model_path = '/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS_score_v3_Exp2/PXS_score_model.pth'
net = models.densenet121()
net.classifier = nn.Linear(1024, 1)
net.load_state_dict(torch.load(model_path))
net.cuda()

def autocrop(image, threshold=1):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    return image

def img_processing(input_image):
    '''
    processes PIL image file -- 
    '''
    output_image = np.array(input_image)
    output_image = Image.fromarray(output_image)

    transf = transforms.Compose([
        transforms.ToTensor()
    ])

    output_image = transf(output_image)

    output_image = np.repeat(output_image, 3, 0)
    output_image = output_image[np.newaxis, ...]
    output_image = Variable(output_image).cuda()

    return output_image

def model_inference(image_path, net):

    img = img_processing(Image.fromarray(image_path))
    net.eval()

    return net(img).item()

def dcm2img2jpg2pxs(dcm_file_path, net, proc_method):
    """Extract the image from a path to a DICOM file."""
    # modified from Jeremy Irvin so as not to use pydicom-gdcm
    # Read the DICOM and extract the image.
    dcm_file = pydicom.dcmread(dcm_file_path, stop_before_pixels=True)

    curr_img, curr_header = load(dcm_file_path)
    raw_image = np.squeeze(curr_img).T.astype(np.float)

    assert len(raw_image.shape) == 2,\
        "Expecting single channel (grayscale) image."

    # # The DICOM standard specifies that you cannot make assumptions about
    # # unused bits in the representation of images, see Chapter 8, 8.1.1, Note 4:
    # # http://dicom.nema.org/medical/dicom/current/output/html/part05.html#chapter_8
    # # pydicom doesn’t exclude those unused bits by default, so we need to mask them
    # raw_image = np.bitwise_and(raw_image, (2 ** (dcm_file.HighBit + 1) -
    #                                        2 ** (dcm_file.HighBit -
    #                                              dcm_file.BitsStored + 1)))

    # Normalize pixels to be in [0, 255].
    raw_image = raw_image - raw_image.min()
    raw_image = raw_image / raw_image.max()
    normalized_image = (raw_image * 255).astype(np.uint8)
 
    # Correct image inversion.
    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        normalized_image = cv2.bitwise_not(normalized_image)

    rescaled_image = autocrop(normalized_image)
    rescaled_image = cv2.resize(rescaled_image, (320, 320), interpolation = cv2.INTER_LINEAR)

    if proc_method == 'histeq':
        # Perform histogram equalization. -- original version
        adjusted_image = cv2.equalizeHist(rescaled_image)

    if proc_method == 'clahe':
        # Perform CLAHE
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8,8))
        adjusted_image = clahe.apply(rescaled_image)

    # jpeg encoding, as per CheXpert/MIMIC
    _, encimg = cv2.imencode('.jpg', adjusted_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) #encode the image using the same preprocessing as chexpert
    decimg = cv2.imdecode(encimg, 0)
    score = model_inference(decimg, net)

    return score



# ## NWH TEST SET ###


annot_table = pd.read_csv('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score/NWH_Covid_Test_Set_noID.csv')
# annot_table = pd.read_excel('/home/home/ken.chang/mnt/2015P002510/Matt/pxs2_analysis/dasa_dataset_summary.xlsx')

PXS_score = []
for i in tqdm(range(len(annot_table))):
    file_path = '/home/home/ken.chang/mnt/2015P002510/Matt/nwh_covid_test_set/' + annot_table.iloc[i].Admission_CXR_Accession + '/IMAGES/' + annot_table.iloc[i].which_image_is_AP
    # file_path = '/home/home/ken.chang/mnt/2015P002510/Matt/dasa_dicoms/' + annot_table.iloc[i].file_name + '.dcm'
    try:
        PXS = dcm2img2jpg2pxs(file_path, net, 'histeq')
        PXS_score.append(PXS)
    except: 
        PXS_score.append('error')
        pass

annot_table['PXS_score'] = PXS_score

annot_table.to_csv('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_nwh_test_nonsiamese_results.csv')

scipy.stats.spearmanr(annot_table['PXS_score'], annot_table['mRALE']) 
scipy.stats.pearsonr(annot_table['PXS_score'], annot_table['mRALE'])  
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(annot_table['PXS_score'], annot_table['mRALE'])
print(r_value)
print(slope)
print(intercept)
print(p_value)

plt.figure()
sns_plot = sns.regplot(annot_table['PXS_score'], annot_table['mRALE'], color = 'black')
sns_plot.set_ylim(0,24)
ylabels = ['{:.0f}'.format(y) for y in sns_plot.get_yticks()]
sns_plot.set_yticklabels(ylabels)
sns_plot.set_xlim(0,24)
plt.xlabel('PXS Score', fontsize = 15) 
plt.ylabel('mRALE Score', fontsize = 15) 
plt.savefig('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_nwh_test_nonsiamese_scatterplot.png')
plt.close()  

 

# ## DASA TEST SET ###


# annot_table = pd.read_csv('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score/NWH_Covid_Test_Set_noID.csv')
annot_table = pd.read_excel('/home/home/ken.chang/mnt/2015P002510/Matt/pxs2_analysis/dasa_dataset_summary.xlsx')

PXS_score = []
for i in tqdm(range(len(annot_table))):
    # file_path = '/home/home/ken.chang/mnt/2015P002510/Matt/nwh_covid_test_set/' + annot_table.iloc[i].Admission_CXR_Accession + '/IMAGES/' + annot_table.iloc[i].which_image_is_AP
    file_path = '/home/home/ken.chang/mnt/2015P002510/Matt/dasa_dicoms/' + annot_table.iloc[i].file_name + '.dcm'
    try:
        PXS = dcm2img2jpg2pxs(file_path, net, 'histeq')
        PXS_score.append(PXS)
    except: 
        PXS_score.append('error')
        pass

annot_table['PXS_score'] = PXS_score

annot_table.to_csv('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_dasa_test_nonsiamese_results.csv')

annot_table = annot_table.rename(columns = {'m_avg':'mRALE'})

scipy.stats.spearmanr(annot_table['PXS_score'], annot_table['mRALE']) 
scipy.stats.pearsonr(annot_table['PXS_score'], annot_table['mRALE'])  
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(annot_table['PXS_score'], annot_table['mRALE'])
print(r_value)
print(slope)
print(intercept)
print(p_value)

plt.figure()
sns_plot = sns.regplot(annot_table['PXS_score'], annot_table['mRALE'], color = 'black')
sns_plot.set_ylim(0,24)
ylabels = ['{:.0f}'.format(y) for y in sns_plot.get_yticks()]
sns_plot.set_yticklabels(ylabels)
sns_plot.set_xlim(0,24)
plt.xlabel('PXS Score', fontsize = 15) 
plt.ylabel('mRALE Score', fontsize = 15) 
plt.savefig('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_dasa_test_nonsiamese_scatterplot.png')
plt.close()   




# ## DASA TEST SET - subset re-rated by MGH raters ###


annot_table = pd.read_csv('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/dasa_mRALE_mgh_raters.csv')

PXS_score = []
for i in tqdm(range(len(annot_table))):
    # file_path = '/home/home/ken.chang/mnt/2015P002510/Matt/nwh_covid_test_set/' + annot_table.iloc[i].Admission_CXR_Accession + '/IMAGES/' + annot_table.iloc[i].which_image_is_AP
    file_path = '/home/home/ken.chang/mnt/2015P002510/Matt/dasa_dicoms/' + annot_table.iloc[i].file_name + '.dcm'
    try:
        PXS = dcm2img2jpg2pxs(file_path, net, 'histeq')
        PXS_score.append(PXS)
    except: 
        PXS_score.append('error')
        pass

annot_table['PXS_score'] = PXS_score

annot_table.to_csv('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_dasa_test_nonsiamese_results_69MGHratersubset.csv')

annot_table = annot_table.rename(columns = {'m_avg':'mRALE_dasa', 'mrale_avg':'mRALE_mgh'})

# scipy.stats.spearmanr(annot_table['PXS_score'], annot_table['mRALE']) 
# scipy.stats.pearsonr(annot_table['PXS_score'], annot_table['mRALE'])  

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(annot_table['PXS_score'], annot_table['mRALE_dasa'])
print(r_value)
print(slope)
print(intercept)
print(p_value)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(annot_table['PXS_score'], annot_table['mRALE_mgh'])
print(r_value)
print(slope)
print(intercept)
print(p_value)

plt.figure()
sns_plot = sns.regplot(annot_table['PXS_score'], annot_table['mRALE_dasa'], color = 'black')
sns_plot.set_ylim(0,24)
ylabels = ['{:.0f}'.format(y) for y in sns_plot.get_yticks()]
sns_plot.set_yticklabels(ylabels)
sns_plot.set_xlim(0,24)
plt.xlabel('PXS Score', fontsize = 15) 
plt.ylabel('mRALE Score', fontsize = 15) 
plt.savefig('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_dasa_test_nonsiamese_scatterplot_69MGHratersubset_dasaratersonly.png')
plt.close()   

plt.figure()
sns_plot = sns.regplot(annot_table['PXS_score'], annot_table['mRALE_mgh'], color = 'black')
sns_plot.set_ylim(0,24)
ylabels = ['{:.0f}'.format(y) for y in sns_plot.get_yticks()]
sns_plot.set_yticklabels(ylabels)
sns_plot.set_xlim(0,24)
plt.xlabel('PXS Score', fontsize = 15) 
plt.ylabel('mRALE Score', fontsize = 15) 
plt.savefig('/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/PXS3_dasa_test_nonsiamese_scatterplot_69MGHratersubset_mghratersonly.png')
plt.close()
