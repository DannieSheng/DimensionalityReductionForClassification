# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:46 2019
Script used to:
    1. Fit knn classifier on the output of the dimensionality reduction by Siamese Network
    2. Generate ROC and precision-recall curves based on the knn result
@author: hdysheng
"""

import pickle
import os
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np
#from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sn
import pdb
plt.close('all')

imgname = '68'
path_gt = r'T:\Results\Analysis CLMB 2018 drone data\grounTruth\Maria_Bradford_Switchgrass_Standplanting\100071_2018_10_31_16_29_53\cropped'
name_gt = 'ground_truth_' + imgname +'.mat'
data_gt = sio.loadmat(os.path.join(path_gt, name_gt))
im_gt   = data_gt['gt']

k           = 5 # k parameter in knn
path_result = r'T:\Results\Analysis CLMB 2018 drone data\orthorectification\Hyperspectral_reflectance\Maria_Bradford_Switchgrass_Standplanting\100071_2018_10_31_16_29_53\parameter1\Siamese\usegt\2\exp3'
pickle_in   = open(os.path.join(path_result, '_results.pkl'), 'rb')
data_dict   = pickle.load(pickle_in)

outputs_train = data_dict['train_output']
labels_train  = np.squeeze(data_dict['train_label'], axis = 1)
index_train   = data_dict['train_index']

outputs_vali = data_dict['vali_output']
labels_vali  = np.squeeze(data_dict['vali_label'], axis = 1)
index_vali   = data_dict['vali_index']

outputs_test = data_dict['test_output']
labels_test  = np.squeeze(data_dict['test_label'], axis = 1)
index_test   = data_dict['test_index']


######## k-nn on outputs
classifier = KNeighborsClassifier(n_neighbors=k)  
classifier.fit(outputs_train, labels_train) 

    ## train
predicted_train     = classifier.predict(outputs_train)
accuracy_c_train    = (predicted_train == labels_train).mean()

confu_train         = confusion_matrix(labels_train, predicted_train, labels=None, sample_weight=None)
df_cm_train         = pd.DataFrame(confu_train)
fig1 = plt.figure()
ax1  = fig1.add_subplot(1,1,1)
sn.heatmap(df_cm_train, annot = True)
plt.title('Confusion matrix for train data')
plt.savefig(os.path.join(path_result, 'cM_train.jpg'))

confu_train_percent = confu_train / confu_train.astype(np.float).sum(axis=1)
fig2 = plt.figure()
ax1  = fig2.add_subplot(1,1,1)
df_cm_train_rc      = pd.DataFrame(confu_train_percent)
sn.heatmap(df_cm_train_rc, annot = True)
plt.title('Normalized confusion matrix for train data')
plt.savefig(os.path.join(path_result, 'cM_train_norm.jpg'))

    ## validation
predicted_vali  = classifier.predict(outputs_vali)
accuracy_c_vali = (predicted_vali == labels_vali).mean()

confu_vali      = confusion_matrix(labels_vali, predicted_vali, labels=None, sample_weight=None)
df_cm_vali      = pd.DataFrame(confu_vali)
fig3 = plt.figure()
ax1  = fig3.add_subplot(1,1,1)
sn.heatmap(df_cm_vali, annot = True)
plt.title('Confusion matrix for validation data')
plt.savefig(os.path.join(path_result, 'cM_vali.jpg'))

confu_vali_percent = confu_vali / confu_vali.astype(np.float).sum(axis=1)
fig4 = plt.figure()
ax1  = fig4.add_subplot(1,1,1)
df_cm_vali_rc      = pd.DataFrame(confu_vali_percent)
sn.heatmap(df_cm_vali_rc, annot = True)
plt.title('Normalized confusion matrix for validation data')
plt.savefig(os.path.join(path_result, 'cM_vali_norm.jpg'))
     
    ## test
predicted_test  = classifier.predict(outputs_test)
accuracy_c_test = (predicted_test == labels_test).mean()

confu_test      = confusion_matrix(labels_test, predicted_test, labels=None, sample_weight=None)
df_cm_test      = pd.DataFrame(confu_test)
fig5 = plt.figure()
ax1  = fig5.add_subplot(1,1,1)
sn.heatmap(df_cm_test, annot = True)
plt.title('Confusion matrix for test data')
plt.savefig(os.path.join(path_result, 'cM_test.jpg'))

confu_test_percent = confu_test / confu_test.astype(np.float).sum(axis=1)
fig6 = plt.figure()
ax1   = fig6.add_subplot(1,1,1)
df_cm_test_rc      = pd.DataFrame(confu_test_percent)
sn.heatmap(df_cm_test_rc, annot = True)
plt.title('Normalized confusion matrix for test data')
plt.savefig(os.path.join(path_result, 'cM_test_norm.jpg'))


####
labels_all = np.concatenate((labels_train, labels_vali, labels_test))
index_all  = np.concatenate((index_train, index_vali, index_test))
label_sorted = labels_all[index_all.argsort()]
index_sorted = index_all[index_all.argsort()]
im_gt_temp = np.reshape(np.zeros(np.shape(im_gt)), (-1,1))
im_gt_temp[index_sorted,:] = np.expand_dims(label_sorted, axis = 1)
im_gt_temp = np.reshape(im_gt_temp, np.shape(im_gt))
fig100 = plt.figure()
ax1  = fig100.add_subplot(1,1,1)
plt.imshow(im_gt_temp)
plt.colorbar()
plt.title('Ground truth image')
plt.savefig(os.path.join(path_result, 'gt.jpg'))


predicted_all = np.concatenate((predicted_train, predicted_vali, predicted_test))
predicted_sorted = predicted_all[index_all.argsort()]
im_predicted = np.reshape(np.zeros(np.shape(im_gt)), (-1,1))
im_predicted[index_sorted,:] = np.expand_dims(predicted_sorted, axis = 1)
im_predicted = np.reshape(im_predicted, np.shape(im_gt))
fig101 = plt.figure()
ax1 = fig101.add_subplot(1,1,1)
plt.imshow(im_predicted)
plt.colorbar()
plt.title('Predicted result image')
plt.savefig(os.path.join(path_result, 'predicted_result_im.jpg'))




######## write the results into a text file
accuracy_list = dict()
accuracy_list['train'] = accuracy_c_train
accuracy_list['validation'] = accuracy_c_vali
accuracy_list['test'] = accuracy_c_test
with open(os.path.join(path_result, 'accuracy_' + str(k) + 'nn.txt'), 'w') as f:
    for key, value in accuracy_list.items():
        f.write(key + ': ' + str(value) + '\n')
f.close()
    
