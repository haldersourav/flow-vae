from IPython import display
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

from network import CVAE
from prepare_dataloader import dataloader_func

from random import randint

from scipy import io
import copy
import pandas as pd

from tqdm import tqdm
from time import sleep
import time

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Loading full dataset avaiable
num_data = 10000                 # number of data files
num_folders = 740                 # number of folders
batch_size = 128                  # batch size
mat_size = 16                     # size of each matrix (square) read

all_data = io.loadmat('../data/sim_data_all.mat')
data = np.asarray(all_data['A_all']).astype('float32')

# Reshape data to make channels first
data_all = np.reshape(data,(num_data,1,mat_size,mat_size))

# Instantiate the network
latent_dim = 3
input_size = 16
model = CVAE(input_size=input_size,latent_dim=latent_dim).to(device)

# Load the model
model.load_state_dict(torch.load('../trained_weights/pytorch_model_weights.pth'))

# Predict on the datset
A_loader = dataloader_func(data_all,100,False)
pbar = tqdm(total=100, position=0, leave=True)
for idx,batch in enumerate(A_loader):
    A = batch.to(device)
    A_pred,_,z_mean,_ = model(A)
    if idx==0:
        A_generated = A_pred.detach().cpu().numpy() 
        data_ls = z_mean.detach().cpu().numpy() 
    elif idx<100:
        A_generated = np.concatenate((A_generated,A_pred.detach().cpu().numpy()) ,axis=0)
        data_ls = np.concatenate((data_ls,z_mean.detach().cpu().numpy()) ,axis=0)
    if idx==99:
        break
    sleep(0.0001)
    pbar.update(1)
    

# Saving predictions on the entire dataset (original)
io.savemat('../data/data_ls_pytorch.mat',{'A_all':data_all,'data_ls':data_ls, 'A_generated':A_generated})