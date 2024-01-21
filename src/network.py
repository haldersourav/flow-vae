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

from random import randint

from scipy import io
import copy
import pandas as pd

from tqdm import tqdm
from time import sleep
import time


class CVAE(nn.Module):
    """Convolutional variational autoencoder."""

    def __init__(self, input_size=16, latent_dim=40):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.input_size*self.input_size*64,128),
            nn.ReLU()         
        )
        
        self.mu = nn.Linear(128,self.latent_dim)
        self.logvar = nn.Linear(128,self.latent_dim)
        self.decoder_pre_process = nn.Sequential(
            nn.Linear(self.latent_dim,self.input_size*self.input_size*64),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # nn.Linear(self.latent_dim,2*self.input_size*self.input_size*64),
            # nn.ReLU(),
            # Unflatten(),#2*self.input_size*self.input_size*64,(2*self.input_size,self.input_size,64)),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding='same')     
        )
        
    def sampling(self, z_mean, z_log_var):       
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + std*eps
    
    def encode(self, X):
        h = self.encoder(X)
        z_mean = self.mu(h)
        z_log_var = self.logvar(h)
        z = self.sampling(z_mean,z_log_var)
        return z, z_mean, z_log_var
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, X):
        z, z_mean, z_log_var = self.encode(X)
        z_preprocessed = self.decoder_pre_process(z)
        decoder_input = z_preprocessed.view(z_preprocessed.size(0), 64, self.input_size,self.input_size)
        out = self.decode(decoder_input)
        return out, z, z_mean, z_log_var