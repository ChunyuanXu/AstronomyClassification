# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:32:58 2018

@author: yuan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 08:36:59 2018

@author: yuan
"""

import pandas as pd
import numpy as np

train_list = pd.read_csv('../first_train_index_20180131.csv')

trainDF_unknown_all = train_list[train_list['type'] == 'unknown']
trainDF_star_all = train_list[train_list['type'] == 'star']

for SAMPLE_NUM in range(16,36):
    # sample star and unknown
    trainDF_unknown = trainDF_unknown_all.sample(frac=0.2, random_state=SAMPLE_NUM)
    trainDF_star = trainDF_star_all.sample(frac=0.02, random_state=SAMPLE_NUM)
    trainDF_unknown.reset_index(drop=True, inplace=True)
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
    np.save('data/train_star_sample'+str(SAMPLE_NUM)+'_int64.npy', train_star)
    
    train_unknown = np.zeros([trainDF_unknown.shape[0], 2601], np.int64)
    for i in range(0,trainDF_unknown.shape[0]):
        trainId_str = str(trainDF_unknown.id[i])
    #    print trainId_str
        txt_file = open('../train/'+trainId_str+'.txt')
        train_feature = map(eval, txt_file.readline().strip().split(','))
        train_unknown[i, :-1] = train_feature
        train_unknown[i, -1] = 3
        txt_file.close()
    print 'unknown', trainDF_unknown.shape[0]
    np.save('data/train_unknown_sample'+str(SAMPLE_NUM)+'_int64.npy', train_unknown)
    print 'Sample number', SAMPLE_NUM, 'done'


