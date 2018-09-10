#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:05:58 2018
@author: codeplay2018
"""

import tensorflow as tf
import networks.tf_utils as tu

def simpnet1(inpt, inpt_size,
                is_training):
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    with tf.name_scope('block1'):
        out = tu.add_conv1d_layer(x_image,32,9,is_training=is_training)
        out = tu.max_pool(out)

    with tf.name_scope('block2'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block3'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block4'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block5'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block6'):
        out = tu.add_fc_layer(out, 128, relu=True, BN=True,
                                        is_training=is_training)
        out = tf.nn.dropout(out, keep_prob)

    with tf.name_scope('block7'):
        y_conv = tu.add_fc_layer(out, 3)
    return y_conv

def simpnet2(inpt, inpt_size,
                is_training):
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    with tf.name_scope('block1'):
        out = tu.add_conv1d_layer(x_image,32,64,is_training=is_training)
        out = tu.max_pool(out)

    with tf.name_scope('block2'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block3'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block4'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block5'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
        
    with tf.name_scope('block6'):
        out = tu.add_fc_layer(out, 128, relu=True, BN=True,
                                        is_training=is_training)
        out = tf.nn.dropout(out, keep_prob)

    with tf.name_scope('block7'):
        y_conv = tu.add_fc_layer(out, 3)
    return y_conv
