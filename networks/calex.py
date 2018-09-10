#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:52:56 2018

@author: codeplay2018
"""
import tensorflow as tf
import networks.tf_utils as tu

def cAlex(inpt, inpt_size,
                is_training):
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first conv-----------------------------
    with tf.name_scope('conv1'):
        conv1_out = tu.add_conv1d_layer(x_image, 16, kernel_size=64, stride=16,
                                        is_training=is_training)
        # output size = INPUT_SIZE/stride_size=256 --->[-1,256,1,num_feature1]
        # this formula is for 'padding="SAME"' situation
    with tf.name_scope('pool1'):
        h_pool1 = tu.max_pool(conv1_out, ksize=3, stride=2, padding='VALID')
        tu.print_activations(h_pool1) 
        # output size: (255-3+1)/2=127(ceiling) --->[-1,127,1,num_feature1]
        
    ## second conv-----------------------------
    with tf.name_scope('conv2'):
        conv2_out = tu.add_conv1d_layer(h_pool1, 32, kernel_size=25,
                                        is_training=is_training)
    with tf.name_scope('pool2'):
        h_pool2 = tu.max_pool(conv2_out, ksize=3, stride=2, padding='VALID')
        tu.print_activations(h_pool2)
        # output size: (127-3+1)/2=63 ---> [-1,63,1,num_feature2]
        
    ## third conv-----------------------------
    with tf.name_scope('conv3'):
        conv3_out = tu.add_conv1d_layer(h_pool2, 32, 9, print_activation=False,
                                        is_training=is_training)
    with tf.name_scope('conv4'):
        conv4_out = tu.add_conv1d_layer(conv3_out, 32, 9, print_activation=False,
                                        is_training=is_training)
    with tf.name_scope('conv5'):
        conv5_out = tu.add_conv1d_layer(conv4_out, 25, 9, print_activation=False,
                                        is_training=is_training)
    with tf.name_scope('pool3'):
        h_pool3 = tu.max_pool(conv5_out, ksize=3, stride=2, padding='VALID')
        tu.print_activations(h_pool3)
        # output size: (63-3+1)/2=31 ---> [-1,31,1,num_feature5]
    
    ## fully connected layers-----------------------------    
    with tf.name_scope('fc1'):
        fc1_out = tu.add_fc_layer(h_pool3, 128, relu=True, BN=True,
                                        is_training=is_training)

    with tf.name_scope('dropout1'):
        h_fc1_drop = tf.nn.dropout(fc1_out, keep_prob)
        
    with tf.name_scope('fc2'):
        fc1_out = tu.add_fc_layer(h_fc1_drop, 128, relu=True, BN=True,
                                        is_training=is_training)
        
    with tf.name_scope('dropout2'):
        h_fc2_drop = tf.nn.dropout(fc1_out, keep_prob)

    with tf.name_scope('fc3'):
        y_conv = tu.add_fc_layer(h_fc2_drop, 3)
    return y_conv