# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 12:18:30 2018

@author: yuan
"""

import pandas as pd
import numpy as np

train_list = pd.read_csv('../first_train_index_20180131.csv')

trainDF_qso = train_list[train_list['type'] == 'qso']
trainDF_galaxy = train_list[train_list['type'] == 'galaxy']

SAMPLE_NUM = 1
# sample star and unknown
trainDF_qso = trainDF_qso.sample(frac=0.2, random_state=SAMPLE_NUM)
trainDF_qso.reset_index(drop=True, inplace=True)

trainDF_galaxy = trainDF_galaxy.sample(frac=0.2, random_state=SAMPLE_NUM)
trainDF_galaxy.reset_index(drop=True, inplace=True)

# load txt data into memory
train_qso = np.zeros([trainDF_qso.shape[0], 2601], np.int64)
for i in range(0,trainDF_qso.shape[0]):
    trainId_str = str(trainDF_qso.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_qso[i,:-1] = train_feature
    train_qso[i, -1] = 0
    txt_file.close()
print 'qso', trainDF_qso.shape[0]
np.save('data/train_qso_sample'+str(SAMPLE_NUM)+'_270.npy', train_qso)

train_galaxy = np.zeros([trainDF_galaxy.shape[0], 2601], np.int64)
for i in range(0,trainDF_galaxy.shape[0]):
    trainId_str = str(trainDF_galaxy.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_galaxy[i,:-1] = train_feature
    train_galaxy[i, -1] = 0
    txt_file.close()
print 'galaxy', trainDF_galaxy.shape[0]
np.save('data/train_galaxy_sample'+str(SAMPLE_NUM)+'_1k.npy', train_galaxy)
