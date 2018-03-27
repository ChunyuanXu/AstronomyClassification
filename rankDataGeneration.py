# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:17:45 2018

@author: yuan
"""

import pandas as pd
import numpy as np

test_csv = pd.read_csv('../first_rank_index_20180307.csv')

test_data = np.zeros([test_csv.shape[0], 2600], np.int64)
for i in range(test_csv.shape[0]):
    test_id = str(test_csv.id[i])
    txt_file = open('../rank/'+test_id+'.txt')
    test_feature = map(eval, txt_file.readline().strip().split(','))
    test_data[i] = test_feature
    txt_file.close()
    if i%10000 == 0:
        print i
np.save('data/rank_data_int64.npy', test_data)