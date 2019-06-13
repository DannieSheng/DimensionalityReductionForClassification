# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:29:12 2019
Script for applying trained Simese model on new data
@author: hdysheng
"""

import os
os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/Python Scripts/DOEdrone/Siamese/SiameseUpdated') # set work directory

import lib.helper_funcs_Siamese_test as lib
import hdf5storage
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import re
import matplotlib.pyplot as plt
import random
import pdb

plt.close('all')
    ## assign parameters
#parameters = lib.parameters
#parameters['modelpath'] = 'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/Siamese/usegt/' + str(parameters['end_dim']) +'/exp' + str(parameters['exp']) + '/'

exp = '3'
end_dim = 110
modelpath  = 'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/Siamese/usegt/' + str(end_dim) +'/exp' + str(exp) + '/'
parameters  = pickle.load(open(os.path.join(modelpath, 'parameters.pkl'), 'rb'))
parameters['modelpath'] = modelpath
parameters['modelname'] = '_model.pth'

## load trained model
model = lib.SiameseNetwork(parameters['inputsize'], parameters['end_dim'])
model.load_state_dict(torch.load(os.path.join(parameters['modelpath'], parameters['modelname'])))

    ## get the list of all files
list_file_temp = [f for f in os.listdir(parameters['hyperpath']) if f.endswith('.mat')]

    ## get the correct order of files
list_file = []
for f in list_file_temp:
    cube_name = re.findall('\d+', f)[0]
    list_file.append(int(cube_name))
index_temp = np.argsort(list_file)
list_file      = [list_file[i] for i in index_temp]
list_file_temp = [list_file_temp[i] for i in index_temp]


    ## use if only care about a sebset of files
if parameters['exp'] is '3': # class 1 and 2
    list_file = [list_file[i] for i in [0, 3, 4, 11, 15]]
elif parameters['exp'] is '4':# class 1 and 3
    list_file = [list_file[i] for i in [0, 3, 4, 7, 11, 12, 15]]
elif parameters['exp'] is '5':# class 2 and 3
    list_file = [list_file[i] for i in [0, 3, 4, 9, 10, 11, 13, 15]]
#    list_file = [list_file[i] for i in [9, 10, 11, 13, 15]]
    
    
output_all = []
label_all  = []

    ## loop over all files
for f in list_file:
    plt.close('all')
    filename = str(f)

    data             = hdf5storage.loadmat(os.path.join(parameters['hyperpath'], 'mappedhyper_'+filename+'.mat'))
    hyper_im         = data['hyper_im']
    flag             = sio.loadmat(os.path.join(parameters['flagpath'], parameters['flagname']), squeeze_me = True)
    goodWvlengthFlag = flag['goodWvlengthFlag']
    hyper_im         = hyper_im[:,:,np.where(goodWvlengthFlag == 1)[0]]
    label            = sio.loadmat(os.path.join(parameters['labelpath'], 'ground_truth_'+filename+'.mat'))
    if parameters['use_gt'] == 1:
        if parameters['use_all_class'] == 1:
            # consider all switchgrass
            map_target = np.ones(np.shape(label['gt']), dtype = int)
            map_target[np.where(label['gt'] == 0)] = 0 # 0 represent for soil&others
            map_target[np.where(label['gt'] == 7)] = 0 # 7 represent for root exclusion zones
        
        else:
            map_target = np.zeros(np.shape(label['gt']), dtype = int)
            map_target[np.where(label['gt'] == parameters['name_class'][0])] = 1
            map_target[np.where(label['gt'] == parameters['name_class'][1])] = 1
        
        hyper_im = hyper_im*map_target[:,:,None]
        label_im = label['gt']*map_target
        plt.imshow(label_im)

    spectra          = np.reshape(hyper_im, [-1, np.shape(hyper_im)[2]])
    gt               = np.reshape(label_im, [-1, 1])

    if parameters['use_gt'] == 1:
        idx_ntarget = np.where(gt == 0)[0]
        spectra     = np.delete(spectra, idx_ntarget, 0)
        gt          = np.delete(gt, idx_ntarget, 0)
    
    idx_all = np.array(range(0, np.shape(label_im)[0]*np.shape(label_im)[1]))
    idx_target = np.setdiff1d(idx_all, idx_ntarget)

    ## visualization of original spectra
    fig300 = plt.figure()
    ax1 = fig300.add_subplot(1,1,1)
    count = 0
    color_code = ['r', 'g']
    rand_idx   =[]
    for i in parameters['name_class']:
        spectra_i = spectra[np.where(gt == i)[0],:]
#        rand_idx = random.sample(range(0, spectra_i.shape[0]), 50)
        idx = random.sample(range(0, spectra_i.shape[0]), 50)
        rand_idx.append(idx)
#        pdb.set_trace()
        temp = np.arange(1, parameters['inputsize']+1)
        ax1.plot(np.expand_dims(temp, axis = 1), np.transpose(spectra_i[idx[0],:]), color_code[count], label = 'class ' + str(parameters['name_class'][count]))
        ax1.plot(np.expand_dims(temp, axis = 1), np.transpose(spectra_i[idx[1:],:]), color_code[count], label = str())
        count += 1

    plt.legend()
    plt.title('Visualization of original spectra')
    plt.savefig(os.path.join(parameters['savepath'], filename + '_spectra.jpg'))

    ## make a transformation on the whole dataset
    data_iter   = lib.create_iterator_single(spectra, gt)
    data_loader = DataLoader(data_iter, batch_size = len(spectra), shuffle = False, num_workers = 0)
    
    
    
    for idx, (inputs, labels) in enumerate(data_loader):
        labels_  = labels.numpy()
        outputs_ = model.forward_once(inputs).detach().numpy()
        output_tosave = {
                'output': outputs_,
                'label': labels_,
                'index_in_im': idx_target}
    

    
    predicted_, output_tosave['accuracy'] = lib.knn_on_output(5, outputs_, idx_target, np.squeeze(labels_, axis = 1), parameters['savepath'], filename)
    f = open(os.path.join(parameters['savepath'], filename+'_results.pkl'), 'wb')
    pickle.dump(output_tosave, f)
    f.close()
    
            ## show the output in image
    label_sorted  = labels[idx_target.argsort()]
    index_sorted  = idx_target[idx_target.argsort()]
    predicted_sorted = predicted_[idx_target.argsort()]
    predicted_sorted = np.expand_dims(predicted_sorted, axis = 1)
    
    im_gt = np.reshape(np.zeros(np.shape(label_im)), (-1,1))
    im_predicted = np.reshape(np.zeros(np.shape(label_im)), (-1,1))
    im_gt[index_sorted, :] =  label_sorted#np.expand_dims(label_sorted, axis = 1)
    im_predicted[index_sorted, :] =  predicted_sorted
    im_gt = np.reshape(im_gt, np.shape(label_im))
    im_predicted = np.reshape(im_predicted, np.shape(label_im))
    fig100 = plt.figure()
    ax1 = fig100.add_subplot(1,2,1)
    plt.imshow(im_gt)
    plt.colorbar()
    plt.title('Ground truth image')
    
    ax2 = fig100.add_subplot(1,2,2)
    plt.imshow(im_predicted)
    plt.colorbar()
    plt.title('Predicted result image')
    plt.savefig(os.path.join(parameters['savepath'], filename + '_predicted_result_im.jpg'))


    ## visualization of output
#for f in list_file:
#    filename = str(f)
#    result_file = open(os.path.join(parameters['savepath'], filename+'_results.pkl'), 'rb')
#    output_dict = pickle.load(result_file)
#    outputs_    = output_dict['output']
#    labels_     = output_dict['label']
    fig1 = plt.figure()
    if parameters['end_dim'] == 3:
        ax1 = fig1.add_subplot(1,1,1, projection = '3d')
    else:
        ax1 = fig1.add_subplot(1,1,1)
    color_code = ['r', 'g']
    count = 0
    for i in parameters['name_class']:
        output_i = outputs_[np.where(labels_ == i)[0],:]
        if parameters['end_dim'] == 2:
            ax1.plot(output_i[:,0], output_i[:,1], '.', label = str(i))
        elif parameters['end_dim'] == 3: 
            ax1.plot(output_i[:,0], output_i[:,1], output_i[:,2], '.', label = str(i))
        else:
            temp = np.arange(1, parameters['end_dim']+1)
            temp = np.expand_dims(temp, axis = 1)
            idx = rand_idx[count]
            ax1.plot(temp, np.transpose(output_i[idx[0],:]), color_code[count], label = 'class ' + str(parameters['name_class'][count]))
            ax1.plot(temp, np.transpose(output_i[idx[1:],:]), color_code[count], label = str())
            count += 1
    ax1.legend()
    plt.title('Visualization of dimensionality reduction rusult')
    plt.savefig(os.path.join(parameters['savepath'], filename + '_visualization.jpg'))

