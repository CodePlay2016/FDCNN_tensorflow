#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:05:58 2018
@author: codeplay2018
"""

import tensorflow as tf
import networks.tf_utils as tu

def speednet1(inpt, inpt_size,
                is_training):
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    end_point={}
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    with tf.name_scope('block1'):
        out = tu.add_conv1d_layer(x_image,32,9,3,is_training=is_training)
        out = tu.max_pool(out, ksize=4)
    end_point['block1'] = out   
    
    with tf.name_scope('block8'):
        out2 = tu.add_fc_layer(out, 128, relu=True, BN=True,
                                        is_training=is_training)
        out2 = tf.nn.dropout(out2, keep_prob)
    with tf.name_scope('block9'):
        out2 = tu.add_fc_layer(out2, 1)

    with tf.name_scope('block2'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out, ksize=4)
    end_point['block2'] = out 
    
    with tf.name_scope('block3'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out, ksize=4)
        
    with tf.name_scope('block4'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out, ksize=4)
        
#    with tf.name_scope('block5'):
#        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
#        out = tu.max_pool(out, ksize=4)
        
    with tf.name_scope('block6'):
        out1 = tu.add_fc_layer(out, 256, relu=True, BN=True,
                                        is_training=is_training)
        out1 = tf.nn.dropout(out1, keep_prob)

    with tf.name_scope('block7'):
        out1 = tu.add_fc_layer(out1, 3)
    
    end_point['class_end'] = out1
    end_point['speed_end'] = out2
    return end_point

def speednet2(inpt, inpt_size,
                is_training):
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    end_point={}
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    with tf.name_scope('block1'):
        out = tu.add_conv1d_layer(x_image,32,64,16,is_training=is_training)
        out = tu.max_pool(out)
    end_point['block1'] = out    

    with tf.name_scope('block2'):
        out = tu.add_conv1d_layer(out,64,9,is_training=is_training)
        out = tu.max_pool(out)
    
    end_point['block2'] = out    
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
        out1 = tu.add_fc_layer(out, 128, relu=True, BN=True,
                                        is_training=is_training)
        out1 = tf.nn.dropout(out1, keep_prob)

    with tf.name_scope('block7'):
        out1 = tu.add_fc_layer(out1, 3)
    
    with tf.name_scope('block8'):
        out2 = tu.add_fc_layer(out, 128, relu=True, BN=True,
                                        is_training=is_training)
        out2 = tf.nn.dropout(out2, keep_prob)
    with tf.name_scope('block9'):
        out2 = tu.add_fc_layer(out2, 1)
    end_point['class_end'] = out1
    end_point['speed_end'] = out2
    return end_point
