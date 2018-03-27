# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 11:04:09 2018

@author: yuan
"""

import pandas as pd
import numpy as np

train_list = pd.read_csv('../first_train_index_20180131.csv')

trainDF_star_all = train_list[train_list['type'] == 'star']

SAMPLE_NUM = 1
# sample star and unknown
trainDF_star = trainDF_star_all.sample(frac=0.2, random_state=SAMPLE_NUM)
trainDF_star.reset_index(drop=True, inplace=True)

# load txt data into memory
train_star = np.zeros([trainDF_star.shape[0], 2601], np.int64)
for i in range(0,trainDF_star.shape[0]):
    trainId_str = str(trainDF_star.id[i])
    txt_file = open('../train/'+trainId_str+'.txt')
    train_feature = map(eval, txt_file.readline().strip().split(','))
    train_star[i,:-1] = train_feature
    train_star[i, -1] = 0
    txt_file.close()
print 'star', trainDF_star.shape[0]
np.save('data/train_star_sample'+str(SAMPLE_NUM)+'_9w.npy', train_star)

