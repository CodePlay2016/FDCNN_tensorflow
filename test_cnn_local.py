#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:31:06 2017

@author: codeplay2017
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals
#
#from builtins import str
## etc., as needed
#
#from future import standard_library
#standard_library.install_aliases()

import time, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow
from sklearn.manifold import TSNE
from data import data_factory
from networks import model_factory

#####-----------structure parameters-----------------
INPUT_SIZE = 8192//4

####-----------------------------------------------------------------------
os.chdir('/media/codeplay2018/545A30205A2FFD74/code/lab/python/FDCNN_tensorflow/')
cwd = os.getcwd()
data_dir = '/media/codeplay2018/545A30205A2FFD74/code/lab/data/TestDataFromWen/'\
            'arranged/steady_condition/pkl/' # ubuntu
model_path = os.path.join(cwd, 'checkpoint/cvgg19/2018-06-07_230020/model.ckpt')
####-----------------------------------------------------------------------
# reader = pywrap_tensorflow.NewCheckpointReader(model_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # Print tensor name and values
# keylist = []
# for key in var_to_shape_map:
#    keylist.append(key)
# list.sort(keylist)
####-----------------------------------------------------------------------
####-----------------------------------------------------------------------

def main(): # _ means the last param
    # Create the model
    Model = model_factory.get_model('cnn')
    model = Model('cvgg19_3', INPUT_SIZE,3)
    
    ### load data
    test_speed = [50]
    batch_size = 1000
    data_fn = data_factory.get_data_from('wen')
    testset,_, _ = data_fn(data_dir,
                            samp_freq='12k',
                            speed_list=test_speed,
                            train_split=0.7,
                            vt_split=0,
                            divide_step=50,
                            data_length = INPUT_SIZE,
                            fft = False,
                            normalize=False,
                            verbose=False, use_speed=True)
    
    num_of_example = testset.num_examples()
    print('number of total test examples is ', num_of_example)
    print('number of test examples is ', batch_size)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        ### test accuracy of model
        test_batch,_ = testset.next_batch(batch_size)
        test_feed = model.get_feed(test_batch, False)
        test_acc, test_sp = sess.run([model.accuracy, model.end_points['speed_end']],
         feed_dict=test_feed)
#        print(test_batch[1])
#        print(test_out)
        print('accuracy is ', test_acc)
        # print(np.concatenate((test_sp,test_batch[2]),axis=1))

    return test_batch#, siglist
            
#########------custom functions----------------------------------
if __name__ == '__main__':
    result = main()
    # pass