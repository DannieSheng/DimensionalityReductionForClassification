# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:10:39 2019
Parameters and helper functions for the Siamese Network for the hyperspectral dataset
@author: hdysheng
"""
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


parameters = {
    'exp': '1',         
#    'filename': ['68', '2068'],
    'filename': ['68'],
    'hyperpath': 'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/mappedWithThermal/mappedHyper/cropped/',
    'flagpath': './data/',
    'flagname': 'flagGoodWvlen.mat',
    'labelpath': 'T:/Results/Analysis CLMB 2018 drone data/grounTruth/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/cropped',
    'use_gt':1,
    'use_all_class':1,
    'end_dim': 110,
    'train_batch_size': 50, # 16, 3, 13
    'train_num_epochs': 200,
    'margin': 1.0,
    'thres_dist': 0.5,
    'learning_rate': 5e-4,
    'name_class': [1,2,3],
    'momentum': 0.9}
if parameters['use_gt'] == 1:
    if parameters['use_all_class'] == 1:
        parameters['savepath'] = 'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/Siamese/usegt/use_all_classes/'+str(parameters['end_dim']) + '/'+'exp'+parameters['exp']
    else:
        parameters['savepath'] = 'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/Siamese/usegt/'+str(parameters['end_dim']) + '/'+'exp'+parameters['exp']
else:
    parameters['savepath'] =  'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/Siamese/useall/'+str(parameters['end_dim']) + '/'+'exp'+parameters['exp']
if not os.path.exists(parameters['savepath']):
    os.makedirs(parameters['savepath'])

    ### helper functions
  ## dataset for dataloader
  # used for paired input
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

  # used for single input
class SiameseDataset_single(Dataset):
    
    def __init__(self, x, label):
        self.size  = label.shape[0]
        self.x    = torch.from_numpy(x)
        self.label = torch.from_numpy(label)
    
    def __getitem__(self, index):
        return(self.x[index],
               self.label[index])
        
    def __len__(self):
        return self.size

	## function to create pairs for Siamese Network    
#def create_pairs(data, digit_indices):
def create_pairs(data, digit_indices, name_class):
    x0_data = []
    x1_data = []
    label   = []

#    n = min([len(digit_indices[d]) for d in range(Config.num_class)])-1 # get the minimum sample size from n classes
    n = min([len(digit_indices[d]) for d in range(len(name_class))])-1 # get the minimum sample size from n classes
#    for d in range(Config.num_class):
    for d in range(len(name_class)):
        for i in range(n):
            
            # generate pairs from same class: label 0
            z0, z1 = digit_indices[d][i], digit_indices[d][i+1]
            # normalization to be added
            x0_data.append(data[z0])
            x1_data.append(data[z1])
            label.append(0)
            
            # generate pairs from different classes: label 1
#            inc    = random.randrange(1,Config.num_class)
            inc    = random.randrange(1,len(name_class))
#            dn     = (d+inc)%Config.num_class
            dn     = (d+inc)%len(name_class)
            z0, z1 = digit_indices[d][i], digit_indices[dn][i]
            x0_data.append(data[z0])
            x1_data.append(data[z1])
            label.append(1)

    x0_data = np.array(x0_data, dtype = np.float32)
#    x0_data = x0_data.reshape([-1,28,28])
    x1_data = np.array(x1_data, dtype = np.float32)
#    x1_data = x1_data.reshape([-1,28,28])
    label   = np.array(label, dtype = np.int32)

    return x0_data, x1_data, label

	## function to create iterable objects 
	# used for paired inputs
def create_iterator(data, label, name_class, shuffle = False):
#def create_iterator(data, label, name_class, shuffle = True):
#    digit_indices = [np.where(label == i)[0] for i in range(Config.num_class)]
    digit_indices = [np.where(label == i)[0] for i in np.nditer(name_class)]
#    x0, x1, label = create_pairs(data, digit_indices)
    x0, x1, label = create_pairs(data, digit_indices, name_class)
    ret           = SiameseDataset(x0, x1, label)
    return ret

    # used for single inputs
def create_iterator_single(data, label, shuffle = False):
#def create_iterator_single(data, label, shuffle = True):
    x  = []
    label_ = []
    for i in range(len(label)):
        x.append(data[i])
        label_.append(label[i])
    x  = np.array(x, dtype = np.float32)
    label_ = np.array(label_, dtype = np.float32)
    ret    = SiameseDataset_single(x, label_)
    return ret

    ## loss function: contrastive loss
class ConstrastiveLoss(nn.Module):
    def __init__(self, margin = parameters['margin']):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_constrastive  = torch.mean((1-label)*torch.pow(euclidean_distance, 2)+
                                        (label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min = 0.0),2))
        return loss_constrastive

    ## Siamese network definition
class SiameseNetwork(nn.Module):
    def __init__(self, inputsize):
        super(SiameseNetwork, self).__init__()
        self.fc1        = nn.Linear(inputsize, 220)
        self.fc2        = nn.Linear(220, 110)
        self.fc3        = nn.Linear(110, 55)
        self.fc4        = nn.Linear(55, 30)
        self.fc5        = nn.Linear(30, 15)
        self.fc6        = nn.Linear(15, 8)
        self.fc7        = nn.Linear(8, 4)
        self.fc8        = nn.Linear(4, parameters['end_dim'])
        self.dropout    = nn.Dropout(0.1)
        
#        self.activation = nn.ReLU(inplace = False)
        self.activation = nn.PReLU()
        
        # forward pass for single input
    def forward_once(self, x):
#        output  = F.relu(self.fc1(x))
#        output  = self.dropout(output)
#        output  = F.relu(self.fc2(output))
#        output  = self.dropout(output)
#        output  = self.fc3(output)
        
        output = self.fc1(x)
        
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc3(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc4(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc5(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc6(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc7(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc8(output)
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

    ## function to compute the accuracy
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances
    '''
    pred = y_pred.ravel() > parameters['thres_dist']
    return np.mean(pred == y_true)

    ## function to generate scatter plots
def scatterplot(outputs, labels, name_class, name_title):
    fig = plt.figure()
    if parameters['end_dim'] !=2:
        ax1 = fig.add_subplot(1,1,1, projection = '3d')
    else:
        ax1 = fig.add_subplot(1,1,1)
    for classname in name_class:
        outputs_i = outputs[np.where(labels == classname)[0],:]
        if parameters['end_dim']!=2:
            ax1.plot(outputs_i[:,0], outputs_i[:,1], outputs_i[:,2],'.', label = str(classname))
        else:
            ax1.plot(outputs_i[:,0], outputs_i[:,1], '.', label = str(classname))
    ax1.legend()
    plt.title('Scatter plot for '+name_title)
    plt.savefig(os.path.join(parameters['savepath'], 'scatter_' + name_title+'.jpg'))
    plt.show() 