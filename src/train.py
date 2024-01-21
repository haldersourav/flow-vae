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
data = np.reshape(data,(num_data,1,mat_size,mat_size))
data_all = copy.deepcopy(data)


# Splitting data into train and test groups
def split_train_test(data, train_size):
    train_data = data[0:train_size,]
    test_data = data[train_size:,]
    return train_data, test_data

train_size = 9600
test_size = num_data - train_size
train_data, test_data = split_train_test(data,train_size)

train_loader = dataloader_func(train_data,train_size,True)
test_loader = dataloader_func(test_data,test_size,True)


# Instantiate the network
latent_dim = 3
input_size = 16
model = CVAE(input_size=input_size,latent_dim=latent_dim).to(device)

def kld_loss(z_mean, z_log_var):
    kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
    kl_loss = torch.mean(kl_loss,1)
    kl_loss *= -0.5
    return torch.mean(kl_loss)

def mse_loss(x, x_pred):
    reconstruction_loss = torch.mean(torch.square(x-x_pred),(1,2,3))
    reconstruction_loss *= 1000
    return torch.mean(reconstruction_loss)

def compute_loss(A, A_pred, z, z_mean, z_log_var, gamma, beta):    
    L1 = mse_loss(A, A_pred)
    L2 = beta*kld_loss(z_mean, z_log_var)
    L = 0.5*L1/gamma + L2
    return L, L1, L2

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
scheduler = StepLR(optimizer, step_size=200, gamma=0.33)

epochs = 250
beta = 1.0
gamma = 0.5
total_loss = []
reconstruction = []
kld = []
for epoch in range(epochs):
    start_time = time.time()
    L = []
    L1 = []
    L2 = []
    for batch in train_loader:
        A = batch.to(device)
        optimizer.zero_grad()
        A_pred, z, z_mean, z_log_var = model.forward(A)
        loss,loss1,loss2 = compute_loss(A, A_pred, z, z_mean, z_log_var, gamma, beta)
        L.append(loss)
        L1.append(loss1)
        L2.append(loss2)
        
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    
    end_time = time.time()
    total_loss.append(torch.mean(torch.Tensor(L)))
    reconstruction.append(torch.mean(torch.Tensor(L1)))
    kld.append(torch.mean(torch.Tensor(L2)))
        
    print('Epoch: {}, Total loss: {}, MSE: {}, KLD: {}, Time: {}'
             .format(epoch, torch.mean(torch.Tensor(L)), torch.mean(torch.Tensor(L1)), torch.mean(torch.Tensor(L2)), end_time-start_time))

# Save the model    
torch.save(model.state_dict(), '../model_weights/pytorch_model_weights.pth')

# Save training losses
total_loss = torch.Tensor(total_loss).detach().cpu().numpy()
reconstruction = torch.Tensor(reconstruction).detach().cpu().numpy()
kld = torch.Tensor(kld).detach().cpu().numpy()
io.savemat('../data/training_losses_pytorch.mat',{'total_loss':total_loss,'reconstruction':reconstruction,'kld':kld})