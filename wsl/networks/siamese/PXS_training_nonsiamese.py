'''
MSE loss implementation of PXS score

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
import pickle
from tqdm import tqdm

# WORKING DIRECTORY
working_path = '/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/'
os.chdir(working_path)

# custom classes
from PXS3_classes_nonsiamese import MGH_Dataset

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='2' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


'''
Load imaging data with annotations
'''

data_table = pd.read_csv('mgh_covid_cxr_data_dicoms_partition.csv')
# data_table = pd.read_csv('mgh_covid_cxr_data_dicoms_partition_clahe.csv')

training_table = data_table[data_table['partition'] == 0]
validation_table = data_table[data_table['partition'] == 1]


# TRAINING DATA 

training_transforms = transforms.Compose([
    transforms.RandomRotation(5), # rotate +/- 5 degrees around center
    transforms.RandomResizedCrop(320, scale=(0.8, 1.2), ratio=(1, 1), interpolation=2),
    transforms.ToTensor()
])

training_dataset = MGH_Dataset(patient_table = training_table,      
                               epoch_size = 1600,
                               transform = training_transforms)

training_dataloader = torch.utils.data.DataLoader(training_dataset, 
                                                  batch_size=16, 
                                                  shuffle=False, 
                                                  num_workers=4)

# VALIDATION DATA 

validation_transforms = transforms.Compose([
    transforms.ToTensor()
])

validation_dataset = MGH_Dataset(patient_table = validation_table,  
                                         epoch_size = 224,
                                         transform = validation_transforms)

validation_dataloader = torch.utils.data.DataLoader(validation_dataset, 
                                                  batch_size=16, 
                                                  shuffle=False, 
                                                  num_workers=4)


""" 
Training the network
"""

def mse_training(training_dataloader, validation_dataloader, working_path, output_folder_name, learning_rate = 0.00002):
    '''
    - Implements MSE loss training
    - Implementation uses early stopping, saving the model with the best validation loss
 
    Arguments
    - training and validation dataloader objects
    - output_folder_name: directory to save model and annotations
    - learning_rate: for Adam optimizer
    ''' 
    net = models.densenet121(pretrained = True)
    net.classifier = nn.Linear(1024, 1) # change last layer to 1 nodes regression output
    net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr = learning_rate)
 
    # Initialization
    num_epochs = 200
    training_losses, validation_losses = [], []

    # Early stopping initialization
    epochs_no_improve = 0
    max_epochs_stop = 7 # "patience" - number of epochs with no improvement in validation loss after which training stops
    validation_loss_min = np.Inf
    history = []

    output_dir = working_path + output_folder_name
    os.mkdir(output_dir)
    f = open(output_dir + "/history_training.txt", 'a+')
    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + "Training starting now...\n")
    f.close()
 
    for epoch in range(0, num_epochs):
        print("epoch training started...")

        # keep track of training and validation loss each epoch
        training_loss = 0
        validation_loss = 0

        # model set to train
        net.train()
          
        # training loop
        for i, data in tqdm(enumerate(training_dataloader, 0)):
            # train neural network
            img0, label, path = data
            img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (DenseNet requires 3-channel input) 
            img0, label = Variable(img0).cuda(), Variable(label).cuda()  # send tensors to the GPU
            optimizer.zero_grad() # clear gradients
            output0 = net(img0)
            loss_mse = criterion(output0[:,-1], label.float())
            loss_mse.backward()
            optimizer.step()

            # keep track of training loss
            training_loss += loss_mse.item()

            print("training loop " + str(i) + " completed")

        else:
            print("validation started...")
            #turn off gradients for validation
            with torch.no_grad(): 
                net.eval() # set evaluation mode

                # determine validation loss
                for j, data2 in tqdm(enumerate(validation_dataloader, 0)):
                    img0, label, path = data2
                    img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (DenseNet requires 3-channel input) 
                    img0, label = Variable(img0).cuda(), Variable(label).cuda()
                    output0 = net(img0)
                    loss_mse = criterion(output0[:,-1], label.float())
                    
                    validation_loss += loss_mse.item()

                    print("validation loop " + str(j) + " completed")

            # calculate average training and validation losses (averaged across batches for the epoch)
            training_loss_avg = training_loss/len(training_dataloader)
            validation_loss_avg = validation_loss/len(validation_dataloader)

            history = [training_losses, validation_losses]
            
            # Save the model if validation loss decreases
            if validation_loss_avg < validation_loss_min:
                # save model
                torch.save(net.state_dict(), output_dir + "/PXS_score_model.pth")
                # track improvement
                epochs_no_improve = 0
                validation_loss_min = validation_loss_avg
                best_epoch = epoch
 
            # Otherwise increment count of epochs with no improvement
            else: 
                epochs_no_improve += 1 
                # Trigger EARLY STOPPING
                if epochs_no_improve >= max_epochs_stop:
                    print(f'\nEarly Stopping! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with loss: {validation_loss_min:.2f}')
                    # Load the best state dict (at the early stopping point)
                    net.load_state_dict(torch.load(output_dir + "/PXS_score_model.pth"))
                    # attach the optimizer
                    net.optimizer = optimizer

                    # save history with pickle
                    with open(output_dir + "/history_training.pckl", "wb") as f:
                        pickle.dump(history, f)

                    f = open(output_dir + "/history_training.txt", 'a+')
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
                            "Early stopping! Total epochs (starting from 0): {:.0f}\n".format(epoch) +
                            "Best epoch: {:.0f}\n".format(best_epoch) +
                            "Validation loss at best epoch: {:.3f}\n".format(validation_loss_min)
                            )
                    f.close()

                    return net, history, output_dir # break the function

        # after each Epoch

        # append to lists for graphing
        training_losses.append(training_loss_avg)
        validation_losses.append(validation_loss_avg)

        print('Training loss : ', training_losses[-1])
        print('Validation loss : ', validation_losses[-1])

        # write history to file
        f = open(output_dir + "/history_training.txt", 'a+')
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
            "Epoch number: {:.0f}\n".format(epoch) +
            "Training loss: {:.3f}\n".format(training_loss_avg) +
            "Validation loss: {:.3f}\n".format(validation_loss_avg) + "\n"
            )
        f.close()
 
    # Load the best state dict (at the early stopping point)
    net.load_state_dict(torch.load(output_dir + "/PXS_score_model.pth"))
    # After training through all epochs attach the optimizer
    net.optimizer = optimizer

    # Return the best model and history
    print(f'\nAll Epochs completed! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with validation loss: {validation_loss_min:.2f}')
    
    return net, history, output_dir

# mse loss training
net, history, output_dir = mse_training(training_dataloader = training_dataloader, 
                                            validation_dataloader = validation_dataloader, 
                                            working_path = '/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score_v3/',
                                            output_folder_name = 'PXS_score_v3_Exp2')

# Training/validation learning curves
plt.title("Number of Training Epochs vs. MSE Loss")
plt.xlabel("Training Epochs")
plt.ylabel("MSE Loss")
plt.plot(range(0, len(history[0])), history[0], label = "Training loss")
plt.plot(range(0, len(history[1])), history[1], label = "Validation loss")
plt.legend(frameon=False)
plt.savefig(output_dir + "/Learning_curve.png")
plt.close()

  

# testing

img0, label, path = next(iter(training_dataloader))

