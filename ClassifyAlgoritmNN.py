#!/usr/bin/env python2
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
import os
import urllib
import collections

from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)              #日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold='nan')                    #打印内容不限制长度

Dataset = collections.namedtuple('Dataset', ['data', 'target'])

# Data sets
data_star = np.load('data/train_star_sample3_int64.npy')
data_galaxy = np.load('data/train_galaxy_int64.npy')
data_qso = np.load('data/train_qso_int64.npy')
data_unknown = np.load('data/train_unknown_sample3_int64.npy')
data_shuffle = np.concatenate((data_star, data_galaxy, data_qso, data_unknown), axis=0)
np.random.shuffle(data_shuffle)
X_train_shuffle, X_val_shuffle, y_train_shuffle, y_val_shuffle = \
train_test_split(data_shuffle[:, :-1], data_shuffle[:, -1], test_size=0.33, random_state=1)
# test_data = np.load('data/test_data_int64.npy')

def main():
        # Load datasets.
        training_set = Dataset(data=X_train_shuffle, target=y_train_shuffle)

        test_set = Dataset(data=X_val_shuffle, target=y_val_shuffle)

        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2600)]

        # Build 3 layer DNN with 10, 20, 10 units respectively.
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=[512, 128, 64, 16], n_classes=4,
                                                    model_dir="/media/yuan/File/astronomyDataAnalysis/code/model2")
        # Define the training inputs
        def get_train_inputs():
                x = tf.constant(training_set.data)
                y = tf.constant(training_set.target)
                return x, y


        # Fit model.
        classifier.fit(input_fn=get_train_inputs, steps=2000)

        # Define the test inputs
        def get_test_inputs():
                x = tf.constant(test_set.data)
                y = tf.constant(test_set.target)

                return x, y

        # Evaluate accuracy.
        #print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
        accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

        print("nTest Accuracy: {0:f}n".format(accuracy_score))

        
        # Classify two new flower samples.
#        def new_samples():
#                return np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
#
        # predictions = list(classifier.predict(input_fn=test_data))
        # np.save('midResult/nn_rst.npy', predictions)
#
#        print("New Samples, Class Predictions:    {}n".format(predictions))

if __name__ == "__main__":
        main()

exit(0)


# model3: 100, 20 nTest Accuracy: 0.693060n 三次迭代后的结果 0.3988 线上效果太差。。

# model: 300, 50, 10 nTest Accuracy: 0.399973n 一次迭代  0.389244n 两次迭代 效果更差了啊

# model1: 300, 50 nTest Accuracy: 0.639549n 第一次迭代

# model2: 512, 128, 64, 16 1:0.401467 2:0.394133
