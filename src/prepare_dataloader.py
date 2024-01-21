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

from random import randint

from scipy import io
import copy
import pandas as pd

from tqdm import tqdm
from time import sleep
import time


class MyDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()       
        A = self.data[idx]
        sample = torch.tensor(A)
        return sample


def dataloader_func(data,batch_size,shuffle_flag=False):
	dataset = MyDataset(data)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag) 

	return data_loader
