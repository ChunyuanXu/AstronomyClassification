# -*- coding:utf-8 -*-
"""
Created on Mon Feb 05 19:49:59 2018

@author: yuan
"""

import pandas as pd
#from enum import Enum
import numpy as np

#class AstronomyType(Enum):
#    star = 1
#    galaxy = 2
#    qso = 3
#    unknown = 4

train_list = pd.read_csv('../first_train_index_20180131.csv')

#for i in range(0,1):
#    trainData_str = str(train_list.id[i])
#    trainData_label = AstronomyType[train_list.type[i]].value
#    txt_file = open('../train/'+trainData_str+'.txt')
#    train_feature = map(eval, txt_file.readline().strip().split(','))
#    txt_file.close()

## 统计训练数据中所有类型的分布情况
#print train_list['type'].value_counts()
#结果
#star       442969
#unknown     34288
#galaxy       5231
#qso          1363

## 尝试将所有测试样本赋值为训练样本中最多的标签
#test_list = pd.read_csv('../first_test_index_20180131.csv')
#test_list['type'] = 'star'
##保存数据结果
#test_list.to_csv('../Submission/Submission_all_star.csv', index=False)

# 要开始用算法解决问题了 hhhhhhhhia~


# idea1: down-sample data to the similar size
# classify data into four parts
trainDF_qso = train_list[train_list['type'] == 'qso']
trainDF_galaxy = train_list[train_list['type'] == 'galaxy']
trainDF_unknown_all = train_list[train_list['type'] == 'unknown']
trainDF_star_all = train_list[train_list['type'] == 'star']

# reset the index of four dataframes
trainDF_qso.reset_index(drop=True, inplace=True)
trainDF_galaxy.reset_index(drop=True, inplace=True)

# sample star and unknown
trainDF_unknown = trainDF_unknown_all.sample(frac=0.2, random_state=1)
trainDF_star = trainDF_star_all.sample(frac=0.02, random_state=1)
trainDF_unknown.reset_index(drop=True, inplace=True)
trainDF_star.reset_index(drop=True, inplace=True)

# load txt data into memory
train_data_shape = trainDF_star.shape[0] + trainDF_galaxy.shape[0] + trainDF_qso.shape[0] + trainDF_unknown.shape[0]
train_data = np.zeros([train_data_shape, 2601], np.uint8)
for i in range(0,trainDF_star.shape[0]):
    trainId_str = str(trainDF_star.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_data[i,:-1] = train_feature
    train_data[i, -1] = 0
    txt_file.close()
print 'star', trainDF_star.shape[0]

#train_galaxy = np.zeros([trainDF_galaxy.shape[0], 2601], np.uint8)
train_data_offset = trainDF_star.shape[0]
for i in range(0,trainDF_galaxy.shape[0]):
    trainId_str = str(trainDF_galaxy.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_data[i+train_data_offset,:-1] = train_feature
    train_data[i+train_data_offset, -1] = 1
    txt_file.close()
print 'galaxy', trainDF_galaxy.shape[0]

#train_qso = np.zeros([trainDF_qso.shape[0], 2601], np.uint8)
train_data_offset = trainDF_star.shape[0] + trainDF_galaxy.shape[0]
for i in range(0,trainDF_qso.shape[0]):
    trainId_str = str(trainDF_qso.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_data[i+train_data_offset,:-1] = train_feature
    train_data[i+train_data_offset, -1] = 2
    txt_file.close()
print 'qso', trainDF_qso.shape[0]

#train_unknown = np.zeros([trainDF_unknown.shape[0], 2601], np.uint8)
train_data_offset = trainDF_star.shape[0] + trainDF_galaxy.shape[0] + trainDF_qso.shape[0]
for i in range(0,trainDF_unknown.shape[0]):
    trainId_str = str(trainDF_unknown.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_data[i+train_data_offset,:-1] = train_feature
    train_data[i+train_data_offset, -1] = 3
    txt_file.close()
print 'unknown', trainDF_unknown.shape[0]

print train_data.shape
np.save('train_data.npy', train_data)

# result record
#star 8859
#galaxy 5231
#qso 1363
#unknown 6858
#(22311, 2601)
# extent idea: increase the number of star and unknown by deal with four category seperately

