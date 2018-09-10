#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:03:10 2017

@author: codeplay2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# etc., as needed

#import os
#os.chdir('/media/codeplay2018/545A30205A2FFD74/code/lab/python/FDCNN_tensorflow/')
#import tensorflow as tf
from data import wen_data
#tf.app.flags.DEFINE_string(
#    'buckets', '/media/codeplay2018/545A30205A2FFD74/code/lab/data/TestDataFromWen/arranged/steady_condition/pkl/',
#    'dataset directory for aliyun setting')
#FLAGS = tf.app.flags.FLAGS

dataset_map = {'wen': wen_data.get_dataset,
                }

def get_data_from(name):
    return dataset_map[name]

    
if __name__ == "__main__":
    get_dataset = get_data_from('wen')
    a,b,c=get_dataset(FLAGS.buckets, speed_list=[10,50], divide_step=20,
                      use_speed=True,verbose=True)
        
    
