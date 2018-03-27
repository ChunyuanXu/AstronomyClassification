# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:49:30 2018

@author: yuan
"""

import numpy as np

predProb_rfc = np.load('midResult/pred_prob_rfc.npy')
predProb_xgb = np.load('midResult/pred_prob_xgb.npy')

predProb = predProb_rfc + predProb_xgb
rst = np.argmax(predProb, axis=1)
np.save('midResult/pred_prob_rfc_xgb.npy', rst)