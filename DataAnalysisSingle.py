# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:35:08 2018

@author: yuan
"""
import pandas as pd
import numpy as np

test_csv = pd.read_csv('../first_test_index_20180131.csv')

test_data = np.zeros([2600])

i = 59739
test_id = str(test_csv.id[i])
print test_id
txt_file = open('../test/'+test_id+'.txt')
test_feature = map(eval, txt_file.readline().strip().split(','))
test_data = np.array(test_feature)
print min(test_data)
print max(test_data)
txt_file.close()