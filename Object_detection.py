#import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


batch_size = 100

# Transform to convert to tensors and Normalize the data
transform = \
    transforms.Compose([transforms.ToTensor(), \
    
    #Data augmentation still needs to be done
    #transforms.HorizontalFlip(probability_goes_here),\
    #transforms.RandomGrayscale(probability_goes_here),\
    
    transforms.Normalize((0.485, 0.456, 0.406), \
                             (0.229, 0.224, 0.225))])

# Load training and testing datasets
train_data = datasets.CIFAR10('data', train=True, 
                              download=True, 
                              transform=transform)
test_data = datasets.CIFAR10('data', train=False,  
                             download=True, 
                             transform=transform)

# Shuffling the training and validation datasets

val_data_size = 0.2
idx = list(range(len(train_data)))

np.random.shuffle(idx)
val_split_index = int(np.floor(val_data_size * len(train_data)))
train_idx, val_idx = idx[val_split_index:], idx[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(train_data, \
                          batch_size=batch_size, \
                          sampler=train_sampler)
val_loader = DataLoader(train_data, \
                        batch_size=batch_size, \
                        sampler=val_sampler)
test_loader = DataLoader(test_data, \
                         batch_size=batch_size)

"""
    Building convolutional neural network
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1, 1)
        self.conv3 = nn.Conv2d(20, 40, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(40 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 40 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.linear2(x), dim=1)
        return x

# Conv1 : Colored image -> 10 filters of size 3. Padding = 1, stride = 1

if __name__ == '__main__':

    print(len(train_idx)) 
