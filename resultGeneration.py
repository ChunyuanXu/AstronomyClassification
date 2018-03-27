#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:43:06 2018

@author: yuan
"""

import numpy as np
import pandas as pd

test_csv = pd.read_csv('../first_rank_index_20180307.csv')
rfc_rst = np.load('midResult/bst_rst_unknown_add2_98_rank.npy')

nameList = ['star', 'galaxy', 'qso', 'unknown']

# all type columns = 'star'
test_csv['type'] = 'star'
# change the type according to the rfc_rst
for i in range(1, 4):
    idx = rfc_rst == i
    test_csv.loc[idx, 'type'] = nameList[i]

print test_csv['type'].value_counts()
# result display
#star       86581
#unknown     9621
#galaxy      3334
#qso          464

test_csv.to_csv('../Submission/Submission_xgb_rst_unknown_add2_98_rank.csv', index=False)
