# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:02:46 2018

@author: yuan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

test_data = np.load('data/test_data_int64.npy')

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2600)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[100, 20], n_classes=4,
                                            model_dir="/media/yuan/File/astronomyDataAnalysis/code/model")

def new_samples():
    return np.array(test_data, dtype=np.float32)

predictions = list(classifier.predict(input_fn=new_samples))
np.save('midResult/nn_rst_model.npy', predictions)

# model3 迭代3次之后
# star       80532
# unknown    12697
# galaxy      5442
# qso         1329
