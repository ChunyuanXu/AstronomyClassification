# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:44:01 2018

@author: yuan
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

NUM_CLASS = 4

data_galaxy = np.load('data/train_galaxy_int64.npy')
data_qso = np.load('data/train_qso_int64.npy')
test_data = np.load('data/test_data_int64.npy')

pred_prob = np.zeros([test_data.shape[0], NUM_CLASS])
for SAMPLE_NUM in range(3, 16):
    data_star = np.load('data/train_star_sample'+str(SAMPLE_NUM)+'_int64.npy')
    data_unknown = np.load('data/train_unknown_sample'+str(SAMPLE_NUM)+'_int64.npy')
    data_shuffle = np.concatenate((data_star, data_galaxy, data_qso, data_unknown), axis=0)
    np.random.shuffle(data_shuffle)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(data_shuffle[:, :-1], data_shuffle[:, -1])
    pred_prob += rfc.predict_proba(test_data)
    print 'SAMPLE_NUM:',SAMPLE_NUM

np.save('midResult/pred_prob_rfc.npy', pred_prob)    
#rfc_combine_rst = np.argmax(pred_prob, axis=1)
#np.save('midResult/rfc_combine_rst_3-16.npy', rfc_combine_rst)
