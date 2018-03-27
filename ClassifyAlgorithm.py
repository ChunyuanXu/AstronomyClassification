#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:21:47 2018

@author: yuan
"""

import numpy as np
from sklearn.model_selection import train_test_split

#data = np.load('train_data.npy')

#data0 = data[0:8859, :-1]
#data1 = data[8859:14090, :-1]
#data2 = data[14090:15453, :-1]
#data3 = data[15453:, :-1]
#
#label0 = data[0:8859, -1]
#label1 = data[8859:14090, -1]
#label2 = data[14090:15453, -1]
#label3 = data[15453:, -1]

#X_train0, X_val0, y_train0, y_val0 = train_test_split(data0, label0, test_size=0.33, random_state=1)
#X_train1, X_val1, y_train1, y_val1 = train_test_split(data1, label1, test_size=0.33, random_state=1)
#X_train2, X_val2, y_train2, y_val2 = train_test_split(data2, label2, test_size=0.33, random_state=1)
#X_train3, X_val3, y_train3, y_val3 = train_test_split(data3, label3, test_size=0.33, random_state=1)
#
#X_train = np.concatenate((X_train0, X_train1, X_train2, X_train3), axis=0)
#y_train = np.concatenate((y_train0, y_train1, y_train2, y_train3), axis=0)
#X_val = np.concatenate((X_val0, X_val1, X_val2, X_val3), axis=0)
#y_val = np.concatenate((y_val0, y_val1, y_val2, y_val3), axis=0)
#print X_train.shape, y_train.shape, X_val.shape, y_val.shape

# METHOD One: try random forest classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
#rfc.fit(X_train, y_train)
#print rfc.score(X_val, y_val)
# result:0.7809911744738629


## Q: Does the order of data matter? try to shuffle the data
#data_shuffle = np.load('train_data.npy')
#np.random.shuffle(data_shuffle)
#X_train_shuffle, X_val_shuffle, y_train_shuffle, y_val_shuffle = train_test_split(data_shuffle[:, :-1], data_shuffle[:, -1],
#                                                                                  test_size=0.33, random_state=1)
#rfc.fit(X_train_shuffle, y_train_shuffle)
#print rfc.score(X_val_shuffle, y_val_shuffle)
# result: 0.7869075105256009

#data_shuffle = np.load('train_data.npy')
data_star = np.load('data/train_star_sample3_int64.npy')
data_galaxy = np.load('data/train_galaxy_int64.npy')
data_qso = np.load('data/train_qso_int64.npy')
data_unknown = np.load('data/train_unknown_sample3_int64.npy')
data_shuffle = np.concatenate((data_star, data_galaxy, data_qso, data_unknown), axis=0)
np.random.shuffle(data_shuffle)
X_train_shuffle, X_val_shuffle, y_train_shuffle, y_val_shuffle = \
train_test_split(data_shuffle[:, :-1], data_shuffle[:, -1], test_size=0.33, random_state=1)
rfc.fit(X_train_shuffle, y_train_shuffle)
pred_rfc = rfc.predict(X_val_shuffle)
from sklearn.metrics import f1_score
rfc_score = f1_score(y_val_shuffle, pred_rfc, average='weighted')
# f1_score result: 0.834758277952
#print rfc.score(X_val_shuffle, y_val_shuffle)
# result: 0.917289148445

# method 1 test data
#rfc.fit(data_shuffle[:, :-1], data_shuffle[:, -1])
#test_data = np.load('data/test_data_int16.npy')
#rfc_rst = rfc.predict(test_data)
#np.save('midResult/rfc_rst_int64.npy', rfc_rst)
