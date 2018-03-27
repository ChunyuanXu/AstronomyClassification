# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:03:15 2018

@author: yuan
"""

import numpy as np
from sklearn.model_selection import train_test_split

data_star = np.load('data/train_star_sample3_int64.npy')
data_galaxy = np.load('data/train_galaxy_int64.npy')
data_qso = np.load('data/train_qso_int64.npy')
data_unknown = np.load('data/train_unknown_sample3_int64.npy')
data_shuffle = np.concatenate((data_star, data_galaxy, data_qso, data_unknown), axis=0)
np.random.shuffle(data_shuffle)

# method Two: XGboost
import xgboost as xgb
X_train_shuffle, X_val_shuffle, y_train_shuffle, y_val_shuffle = \
train_test_split(data_shuffle[:, :-1], data_shuffle[:, -1], test_size=0.1, random_state=1)
dtrain = xgb.DMatrix(X_train_shuffle, label=y_train_shuffle)
dval = xgb.DMatrix(X_val_shuffle, label=y_val_shuffle)
param = {'objective':'multi:softmax', 'max_depth':9, 'eta':0.2, 'silent':1,  \
        'num_class':4, 'seed':1, 'min_child_weight':4, 'gamma':0.5, 'scale_pos_weight':1}
num_round = 20
bst = xgb.train(param, dtrain, num_round)
pred = bst.predict(dval)

from sklearn.metrics import f1_score
xgb_score = f1_score(y_val_shuffle, pred, average='weighted')
print xgb_score
# result: 0.780217562168

# train_data = xgb.DMatrix(data_shuffle[:, :-1], label=data_shuffle[:, -1])
# dtest = np.load('data/test_data_int64.npy')
# test_data = xgb.DMatrix(dtest)
# param = {'objective':'multi:softmax', 'max_depth':9, 'eta':0.2, 'silent':1,  \
#        'num_class':4, 'seed':1, 'min_child_weight':4, 'gamma':0.5, 'scale_pos_weight':1}
# num_round = 50
# bst = xgb.train(param, train_data, num_round)
# bst_rst = bst.predict(test_data)
# np.save('midResult/bst_rst_10w.npy', bst_rst)
