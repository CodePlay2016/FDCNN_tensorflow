#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:15:22 2018

@author: codeplay2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import glob, os, pickle

data_dir = '/home/codeplay2018/Desktop/Link to lab/data/TestDataFromWen/arranged/steady_condition/'

file_list = glob.glob(os.path.join(data_dir,'*.txt'))

#for file in file_list:
#    data = np.loadtxt(file)
#    print(data.shape)
#    filename = file.split('/')[-1].split('.')[0]
#    with tf.gfile.GFile(data_dir+'pkl/'+filename+'.pkl', 'wb') as pf:
#        pickle.dump(data, pf)
    
with open(data_dir+'pkl/na_20k_10.pkl','rb') as f:
    print('load')
    a = pickle.load(f)
    print('over')
print(a.shape)

