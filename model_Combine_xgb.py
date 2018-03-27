# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:15:47 2018

@author: yuan
"""

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

NUM_CLASS = 4

data_galaxy = np.load('data/train_galaxy_int64.npy')
data_qso = np.load('data/train_qso_int64.npy')
dtest = np.load('data/test_data_int64.npy')
test_data = xgb.DMatrix(dtest)

param = {'objective':'multi:softprob', 'max_depth':9, 'eta':0.02, 'silent':1,  \
        'num_class':4, 'seed':1, 'min_child_weight':4, 'gamma':0.5, 'scale_pos_weight':1}
num_round = 100

print 'xgb start'
pred_prob = np.zeros([dtest.shape[0], NUM_CLASS])
for SAMPLE_NUM in range(3, 36):
    data_star = np.load('data/train_star_sample'+str(SAMPLE_NUM)+'_int64.npy')
    data_unknown = np.load('data/train_unknown_sample'+str(SAMPLE_NUM)+'_int64.npy')
    data_shuffle = np.concatenate((data_star, data_galaxy, data_qso, data_unknown), axis=0)
    np.random.shuffle(data_shuffle)
    
    train_data = xgb.DMatrix(data_shuffle[:, :-1], label=data_shuffle[:, -1])
    
    bst = xgb.train(param, train_data, num_round)
    pred_prob += bst.predict(test_data)
    print 'Sample Number:', SAMPLE_NUM

np.save('midResult/pred_prob_xgb_3-36.npy', pred_prob)
xgb_combine_rst = np.argmax(pred_prob, axis=1)
np.save('midResult/xgb_combine_rst_3-36.npy', xgb_combine_rst)
