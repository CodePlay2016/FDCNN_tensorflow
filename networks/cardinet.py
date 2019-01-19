'''
This network refers the paper 
'Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks'
'''
# -*- coding: utf-8 -*-
import tensorflow as tf
import networks.tf_utils as tu

def first_block(inpt, num_features,kernel_size,is_training,dropout_rate=0.8):
    a, b = tf.constant(dropout_rate,dtype=tf.float32), tf.constant(1.0,dtype=tf.float32)
    dropout_rate = tf.cond(is_training, lambda: a, lambda: b)
    out = tf.layers.conv1d(inpt,num_features,kernel_size,padding='SAME')
    out = tf.layers.batch_normalization(out, training=is_training)
    out1 = tf.nn.relu(out)

    out = tf.layers.conv1d(out1,num_features,kernel_size,padding='SAME')
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out,rate=dropout_rate)
    out = tf.layers.conv1d(out,num_features,kernel_size,strides=2,padding='SAME')

    out1 = tf.layers.max_pooling1d(out1,kernel_size,2,'SAME')
    out = out + out1
    return out

def build_block(inpt, num_features, kernel_size, down_sample, is_training, dropout_rate=0.5):
    a, b = tf.constant(dropout_rate,dtype=tf.float32), tf.constant(1.0,dtype=tf.float32)
    dropout_rate = tf.cond(is_training, lambda: a, lambda: b)
    pre_num_features = inpt.get_shape().as_list()[-1]
    stride = 2 if down_sample else 1

    out = tf.layers.batch_normalization(inpt, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out,dropout_rate)
    out = tf.layers.conv1d(out,num_features,1,padding='SAME')
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out,dropout_rate)
    out = tf.layers.conv1d(out,num_features,kernel_size,strides=stride,padding='SAME')

    # shortcut
    inpt = tf.layers.max_pooling1d(inpt,kernel_size,stride,'SAME')
    if not num_features == pre_num_features:
        inpt = tf.layers.conv1d(inpt,num_features,1)
    out = out + inpt
    return out

def final_block(inpt,num_class, is_training):
    out = tf.layers.batch_normalization(inpt, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.flatten(out)
    out1 = tf.layers.dense(out, num_class)
    out2 = tf.layers.dense(out, 1)
    end_points = {'class_end':out1,'speed_end': out2}
    return end_points

def cardinet(inpt,_,is_training):
    num_features = 64
    kernel_size = 16
    num_build_blocks = 15
    feature_increase_each_n_block = 4
    downsample_each_n_block = 2
    
    inpt = tf.expand_dims(inpt,-1)

    out = first_block(inpt, num_features, kernel_size, is_training, dropout_rate=0.8)

    for ii in range(num_build_blocks):
        if ii % feature_increase_each_n_block == 0:
            num_features *= 2
        if ii % downsample_each_n_block == 0:
            downsample = True
        else: downsample = False
        out = build_block(out, num_features, kernel_size, downsample, is_training, dropout_rate=0.8)
    
    out = final_block(out, 3, is_training)

    return out



    