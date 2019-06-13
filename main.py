# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:01:01 2019
Script of applying Siamese Network on DOE hyperspectral dataset
Reference:
    https://github.com/delijati/pytorch-siamese/blob/master/train_mnist.py
@author: hdysheng
"""
import torch
import os
os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/Python Scripts/DOEdrone/Siamese/SiameseUpdated')
#import random
import numpy as np
from torch.utils.data import DataLoader
import lib.helper_funcs_Siamese as lib
import hdf5storage
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import pdb

plt.close('all')
    ## training and test data set (load .mat files)
parameters       = lib.parameters


spectra_ = []
gt_      = []
#spectra2 = np.array([])
#gt2      = np.array([])
for file in parameters['filename']:
    data             = hdf5storage.loadmat(os.path.join(parameters['hyperpath'], 'mappedhyper_'+file+'.mat'))
    hyper_im         = data['hyper_im']
    flag             = sio.loadmat(os.path.join(parameters['flagpath'], parameters['flagname']), squeeze_me = True)
    goodWvlengthFlag = flag['goodWvlengthFlag']
    hyper_im         = hyper_im[:,:,np.where(goodWvlengthFlag == 1)[0]]
    label            = sio.loadmat(os.path.join(parameters['labelpath'], 'ground_truth_'+file+'.mat'))
#    gt               = label['gt']
    
    if parameters['use_gt'] == 1:
        if parameters['use_all_class'] == 1:
            # consider all switchgrass
            map_target = np.ones(np.shape(label['gt']), dtype = int)
            map_target[np.where(label['gt'] == 0)] = 0 # 0 represent for soil&others
            map_target[np.where(label['gt'] == 7)] = 0 # 7 represent for root exclusion zones
        
        else:
            # consider only 2 classes of switchgrass
            map_target = np.zeros(np.shape(label['gt']), dtype = int)
            map_target[np.where(label['gt'] == parameters['name_class'][0])] = 1
            map_target[np.where(label['gt'] == parameters['name_class'][1])] = 1
        
        hyper_im = hyper_im*map_target[:,:,None]
        label_im = label['gt']*map_target
        plt.imshow(label_im)
        plt.colorbar()
    temp_spectra = np.reshape(hyper_im, [-1, np.shape(hyper_im)[2]])  
    temp_gt      = np.reshape(label_im, [-1, 1])
    if parameters['use_gt'] == 1:
        idx_ntarget  = np.where(temp_gt == 0)[0]
        temp_spectra = np.delete(temp_spectra, idx_ntarget, 0)
        temp_gt      = np.delete(temp_gt, idx_ntarget, 0)
    idx_all    = np.array(range(0, np.shape(label_im)[0]* np.shape(label_im)[1]))
    idx_target = np.setdiff1d(idx_all, idx_ntarget)
    spectra_.append(temp_spectra)
    gt_.append(temp_gt)
    
    # only pixels in one file are used for training         
spectra = spectra_[0]
gt      = gt_[0]

#    # only pixels in two files are used for training            
#spectra = np.concatenate((spectra_[0], spectra_[1]), axis = 0)
#gt      = np.concatenate((gt_[0], gt_[1]), axis = 0)


parameters['name_class'] = np.unique(gt)
parameters['num_class']  = len(np.unique(gt))
parameters['inputsize']  = np.shape(spectra)[1]


    ## save the parameters in a txt file
#with open(os.path.join(parameters.savepath, 'parameters.txt'), 'w') as f:
#    for attr, value in parameters.__dict__.items():
#        if not attr.startswith('__'):
#            f.write(attr + ': ' + str(value) + '\n')
#f.close()
    
with open(os.path.join(parameters['savepath'], 'parameters.txt'), 'w') as f:
    for key, value in parameters.items():
        f.write(key + ': ' + str(value) + '\n')
f.close()

f = open(os.path.join(parameters['savepath'], 'parameters.pkl'), 'wb')
pickle.dump(parameters, f)
f.close()

    ## train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(spectra, gt, idx_target, test_size = 0.2, random_state = 0)

    ## train-validation split
X_train, X_vali, y_train, y_vali, idx_train, idx_vali = train_test_split(X_train, y_train, idx_train, test_size = 0.2, random_state = 0)

    ## model
model = lib.SiameseNetwork(parameters['inputsize'])

    ## loss function and optimizer
criterion = lib.ConstrastiveLoss()

    # SGD
#optimizer  = torch.optim.SGD(model.parameters(), lr = parameters.learning_rate, momentum = parameters.momentum)
    # Adam
optimizer = torch.optim.Adam(model.parameters(), lr = parameters['learning_rate'])

    ## training
train_iter   = lib.create_iterator(X_train, y_train, parameters['name_class'])
vali_iter    = lib.create_iterator(X_vali, y_vali, parameters['name_class'])
#pdb.set_trace()
train_loader = DataLoader(train_iter, batch_size = parameters['train_batch_size'], shuffle = True, num_workers = 0)
vali_loader  = DataLoader(vali_iter, batch_size = vali_iter.size, shuffle = True, num_workers = 0)
model.train()

    # loss after every training batch
train_loss_hist  = []
train_accu_hist  = []
avg_dist_same    = []
avg_dist_diff    = []
std_dist_same    = []
std_dist_diff    = []
avg_dist_same_vali    = []
avg_dist_diff_vali    = []
std_dist_same_vali    = []
std_dist_diff_vali    = []

    # loss after each epoch
train_loss_all = []
train_accu_all = []
vali_loss_all  = []
vali_accu_all  = []

counter          = []

dist_train       = []
dist_vali        = []
iteration_number = 0
for idx_epoch in range(0, parameters['train_num_epochs']):
    loss_temp = 0
    accu_temp = 0
    for idx_batch, (x0, x1, trainlabel) in enumerate(train_loader):
        trainlabel = trainlabel.float()
        output0, output1 = model(x0, x1)
        loss             = criterion(output0, output1, trainlabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_hist.append(loss.item())
        loss_temp += loss.item()
        distances = model.predict(output0, output1)
        accu      = lib.compute_accuracy(trainlabel.detach().numpy(), distances.detach().numpy())
        train_accu_hist.append(accu)
        accu_temp += accu
        avg_dist_same.append(np.mean(distances.detach().numpy()[np.where(trainlabel.detach().numpy() == 0)]))
        avg_dist_diff.append(np.mean(distances.detach().numpy()[np.where(trainlabel.detach().numpy() == 1)]))
        std_dist_same.append(np.std(distances.detach().numpy()[np.where(trainlabel.detach().numpy() == 0)]))
        std_dist_same.append(np.std(distances.detach().numpy()[np.where(trainlabel.detach().numpy() == 1)]))
        if idx_batch%100 == 0:
            print('=========================================================')
            print('Epoch number {}\n Current loss {}\n'.format(idx_epoch+1, loss.item()))
            print('Current accuracy {}\n'.format(accu))
            iteration_number += 100
            counter.append(iteration_number)
#        if idx_epoch == parameters['train_num_epochs']-1:
#            dist_train.append(distances.detach().numpy)
    
    train_loss_all.append(loss_temp/(idx_batch+1))
    train_accu_all.append(accu_temp/(idx_batch+1))
#    pdb.set_trace()
    
    # apply the model on the validation set after each batch
    for idx, (x0, x1, valilabels) in enumerate(vali_loader):
        valilabels = valilabels.float()
        outputs0, outputs1 = model(x0, x1)
        distances_vali     = model.predict(outputs0, outputs1)
        avg_dist_same_vali.append(np.mean(distances_vali.detach().numpy()[np.where(valilabels.detach().numpy() == 0)]))
        avg_dist_diff_vali.append(np.mean(distances_vali.detach().numpy()[np.where(valilabels.detach().numpy() == 1)]))
        std_dist_same_vali.append(np.std(distances_vali.detach().numpy()[np.where(valilabels.detach().numpy() == 0)]))
        std_dist_same_vali.append(np.std(distances_vali.detach().numpy()[np.where(valilabels.detach().numpy() == 1)]))
#        if idx == (parameters['train_num_epochs']-1):
        if idx_epoch == (parameters['train_num_epochs']-1): # only save the distances for the last training epoch as the validation distances
#            pdb.set_trace()
            dist_vali.append(distances_vali.detach().numpy())
        vali_accuracy = lib.compute_accuracy(valilabels.detach().numpy(), distances_vali.detach().numpy())
#        vali_loss     = torch.mean((1-valilabels)*torch.pow(distances_vali, 2)+
#                                        (valilabels)*torch.pow(torch.clamp(parameters['margin']-distances_vali, min = 0.0),2))
        vali_loss      = criterion(outputs0, outputs1, valilabels).item()
    vali_accu_all.append(vali_accuracy)
    vali_loss_all.append(vali_loss)
    

    ## save the final model
torch.save(model.state_dict(), os.path.join(parameters['savepath'], '_model.pth'))
#pdb.set_trace()
    ## ROC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(valilabels, dist_vali[0], drop_intermediate = True)
    ## precision-recall
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(valilabels, dist_vali[0])


fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)   
ax.plot(train_loss_hist, label = 'train loss')
ax.plot(train_accu_hist, label = 'train accuracy')    
ax.legend(loc = 'upper right')  
ax.set_ylim([0, 1])  
plt.xlabel('training batch')
plt.ylabel('loss or accuracy')   
plt.savefig(os.path.join(parameters['savepath'], '_train.jpg'))
plt.show()

fig2 = plt.figure()
ax   = fig2.add_subplot(1,1,1)
ax.plot(train_loss_all, label = 'train loss')
ax.plot(train_accu_all, label = 'train accuracy')
ax.plot(vali_loss_all, label = 'validation loss')
ax.plot(vali_accu_all, label = 'validation accuracy')
ax.legend(loc = 'upper right')
ax.set_ylim([0, 1])  
plt.xlabel('training epoch')
plt.ylabel('loss or accuracy')  
plt.savefig(os.path.join(parameters['savepath'], '_train_epoch.jpg'))
plt.show()  

fig3 = plt.figure()
ax   = fig3.add_subplot(1,1,1)
ax.plot(avg_dist_diff, label = 'points in differences classes')
ax.plot(avg_dist_same, label = 'points in same class')
plt.legend(loc = 'upper right')
plt.xlabel('training batch')
plt.ylabel('average distance') 
plt.savefig(os.path.join(parameters['savepath'], '_average_distances.jpg'))
plt.show()

fig4 = plt.figure()
ax1   = fig4.add_subplot(1,2,1)
ax1.plot(fpr, tpr, label = 'ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve on pairs of validation data')

ax2   = fig4.add_subplot(1,2,2)
ax2.plot(recall, precision, label = 'PrecisionRecall')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('Precision Recall curve on pairs of validation data')
plt.savefig(os.path.join(parameters['savepath'], '_ROC_PR.jpg'))
plt.show()

    ## test: test all at a time
test_iter   = lib.create_iterator(X_test, y_test, parameters['name_class'])
test_loader = DataLoader(test_iter, batch_size = test_iter.size , shuffle = True, num_workers = 0)

for idx, (x0, x1, labels) in enumerate(test_loader):
    labels = labels.float()
    outputs0, outputs1 = model(x0, x1)
    distances          = model.predict(outputs0, outputs1)
    test_accuracy      = lib.compute_accuracy(labels.detach().numpy(), distances.detach().numpy())
    
print('=========================================================')
print('Test accuracy {}\n'.format(test_accuracy.item()))

    ## make a transformation on the whole dataset
train_iter_all = lib.create_iterator_single(X_train, y_train)
vali_iter_all  = lib.create_iterator_single(X_vali, y_vali)
test_iter_all  = lib.create_iterator_single(X_test, y_test)
train_loader_all = DataLoader(train_iter_all, batch_size = len(X_train), shuffle = False, num_workers = 0)
vali_loader_all  = DataLoader(vali_iter_all, batch_size = len(X_vali), shuffle = False, num_workers = 0)
test_loader_all  = DataLoader(test_iter_all, batch_size = len(X_test), shuffle = False, num_workers = 0)
for idx, (inputs_train, labels_train) in enumerate(train_loader_all):
    labels_train  = labels_train.numpy()
    outputs_train = model.forward_once(inputs_train).detach().numpy()
    
for idx, (inputs_vali, labels_vali) in enumerate(vali_loader_all):
    labels_vali  = labels_vali.numpy()
    outputs_vali = model.forward_once(inputs_vali).detach().numpy()
    
for idx, (inputs_test, labels_test) in enumerate(test_loader_all):
    labels_test  = labels_test.numpy()
    outputs_test = model.forward_once(inputs_test).detach().numpy()
outputs_all = np.concatenate((outputs_train, outputs_vali, outputs_test), axis = 0)
labels_all  = np.concatenate((labels_train, labels_vali, labels_test), axis = 0)

#pdb.set_trace()

output_tosave = {
        'all_output': outputs_all,
        'all_label': labels_all,
        'train_output': outputs_train,
        'train_label': labels_train,
        'vali_output': outputs_vali,
        'vali_label': labels_vali,
        'test_output': outputs_test,
        'test_label': labels_test,
        'fpr': fpr, 
        'tpr': tpr, 
        'precision': precision,
        'recall': recall,
        'train_index': idx_train,
        'vali_index': idx_vali,
        'test_index': idx_test}
f = open(os.path.join(parameters['savepath'], '_results.pkl'), 'wb')
pickle.dump(output_tosave, f)
f.close()

sio.savemat(os.path.join(parameters['savepath'], '_results.mat'), output_tosave)

#import pickle
#import os
#os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/Python Scripts/DOEdrone/Siamese/SiameseUpdated')
#
#import lib.helper_funcs_Siamese as lib
#parameters = lib.parameters
#pickle.load(os.path.join(parameters['savepath'], parameters['filename']+'_results.pkl'))

lib.scatterplot(outputs_all, labels_all, parameters['name_class'], 'all data')
lib.scatterplot(outputs_train, labels_train, parameters['name_class'], 'train data')
lib.scatterplot(outputs_vali, labels_vali, parameters['name_class'], 'vali data')
lib.scatterplot(outputs_test, labels_test, parameters['name_class'], 'test data')
