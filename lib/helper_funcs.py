# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:56:57 2019
helper functions for the Siamese Network
References: 
    https://github.com/delijati/pytorch-siamese/blob/master/train_mnist.py
@author: hdysheng
"""
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

    ## Configuration
class Config():
    train_batch_size = 100
    test_batch_size  = 1
    train_num_epochs = 200
    num_class        = 10
    margin           = 1.0
    thres_dist       = 1.25
    learning_rate    = 0.005
    momentum         = 0.9


    ### helper functions
    ## dataset for dataloader
class SiameseDataset(Dataset):
    
    def __init__(self, x0, x1, label):
        self.size  = label.shape[0]
        self.x0    = torch.from_numpy(x0)
        self.x1    = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)
    
    def __getitem__(self, index):
        return(self.x0[index],
               self.x1[index],
               self.label[index])
        
    def __len__(self):
        return self.size

	## function to create pairs for Siamese Network
def create_pairs(data, digit_indices):
    x0_data = []
    x1_data = []
    label   = []
    
    n = min([len(digit_indices[d]) for d in range(Config.num_class)])-1 # get the minimum sample size from 10 classes
    for d in range(Config.num_class):
        for i in range(n):
            
            # generate pairs from same class: label 0
            z0, z1 = digit_indices[d][i], digit_indices[d][i+1]
            # normalization, can be done at data loading maybe
            x0_data.append(data[z0]/255.0)
            x1_data.append(data[z1]/255.0)
            label.append(0)
            
            # generate pairs from different classes: label 1
            inc    = random.randrange(1,Config.num_class)
            dn     = (d+inc)%Config.num_class
            z0, z1 = digit_indices[d][i], digit_indices[dn][i]
            x0_data.append(data[z0]/255.0)
            x1_data.append(data[z1]/255.0)
            label.append(1)
            
            
    x0_data = np.array(x0_data, dtype = np.float32)
#    x0_data = x0_data.reshape([-1,1,28,28])
    x0_data = x0_data.reshape([-1,28,28])
    x1_data = np.array(x1_data, dtype = np.float32)
#    x1_data = x1_data.reshape([-1,1,28,28])
    x1_data = x1_data.reshape([-1,28,28])
    label   = np.array(label, dtype = np.int32)
    return x0_data, x1_data, label

def create_iterator(data, label, shuffle = True):
    digit_indices = [np.where(label == i)[0] for i in range(Config.num_class)]
    x0, x1, label = create_pairs(data, digit_indices)
    ret           = SiameseDataset(x0, x1, label)
    return ret

	## definition of loss function: constrastive loss
class ConstrastiveLoss(nn.Module):
    def __init__(self, margin = Config.margin):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_constrastive  = torch.mean((1-label)*torch.pow(euclidean_distance, 2)+
                                        (label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min = 0.0),2))
        return loss_constrastive

	## definition of network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout    = nn.Dropout(0.1)
        
        # forward pass for single input
    def forward_once(self, x):
        output  = x.view(-1,784)
        output  = F.relu(self.fc1(output))
        output  = self.dropout(output)
        output  = F.relu(self.fc2(output))
        output  = self.dropout(output)
        output  = self.fc3(output)
        return output
    
        # forward pass for the whole network
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
        
        # predict 
    def predict(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return euclidean_distance

    ## function to compute classification accuracy  
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances
    '''
    pred = y_pred.ravel() > Config.thres_dist
    return np.mean(pred == y_true)

def scatterplot(outputs, labels, numclass, name_title):
    fig = plt.figure()
#    ax1 = fig.add_subplot(1,1,1)
#    ax1.set_xlim([-1,3])
#    ax1.set_ylim([-1,1])
    ax2 = fig.add_subplot(1,1,1)
    ax2.set_xlim([-1,3])
    ax2.set_ylim([-1,1])
    for iclass in range(0,numclass):
        outputs_i = outputs[np.where(labels == iclass)]
#        ax1.scatter(outputs_i[:,0], outputs_i[:,1], linewidth = 0.5, alpha = 0.6, label = str(iclass))
        ax2.plot(outputs_i[:,0], outputs_i[:,1], '.', label = str(iclass))
    plt.title('Scatter plot for '+name_title)
    plt.legend()
#    save
    plt.show()