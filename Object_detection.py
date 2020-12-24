import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy.io as io
from numpy import savetxt

def load_datasets(val_data_size = 0.2, batch_size = 100):

    # Transform to convert to tensors, Data augmentation and Normalize the data
    transform = \
    transforms.Compose([transforms.RandomHorizontalFlip(0.5),\
                        transforms.RandomGrayscale(0.1),\
                        transforms.ToTensor(), \
                        transforms.Normalize((0.5, 0.5, 0.5), \
                                            (0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR10('data', train=True, 
                                download=True, 
                                transform=transform)

    vocab = torch.load("mnist/MNIST/processed/training.pt")
    print("PT samples", vocab)

    # Load training and testing datasets
    mat = io.loadmat('mnist.mat')
   
    tx_data = mat['trainX']
    tx_target = mat['trainY']

    tx_target = torch.from_numpy(tx_target)

    tx_data = np.reshape(tx_data, (60000, 28, 28))
    tx_data = torch.from_numpy(tx_data)

    """
    sample = []
    sample.append(tx_data)
    sample.append(tx_target)
    """

    #print("After stack samples", samples)

    #print("sample train", np.shape(sample[0]))
    #print("sample target", np.shape(sample[1]))

    #train_data = datasets.MNIST(root='training.pt', train=True, transform=transform)

    tx_data = TensorDataset(tx_data)

    tx_data.transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5), \
                        transforms.RandomGrayscale(0.1), \
                        transforms.ToTensor(), \
                        transforms.Normalize((0.5, 0.5, 0.5), \
                                            (0.5, 0.5, 0.5))])
    tx_target = TensorDataset(tx_target)

    samples = (tx_data, tx_target)    

    print("Samples", samples)

    """
    ty_data = mat['testX']
    ty_target = mat['testY']
    
    ty_data = torch.from_numpy(ty_data).float()
    ty_target = torch.from_numpy(ty_target).long() 
    ty = {ty_data, ty_target}
    ty_dataset = TensorDataset(ty)
    """

if __name__ == '__main__':
    load_datasets()

