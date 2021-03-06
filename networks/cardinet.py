'''
This network refers the paper 
'Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks'
'''
# -*- coding: utf-8 -*-
import tensorflow as tf
import networks.tf_utils as tu

def first_block(inpt, num_features,kernel_size,is_training,dropout_rate=0.8):
    out = tf.layers.conv1d(inpt,num_features,kernel_size,padding='SAME')
    out = tf.layers.batch_normalization(out, training=is_training)
    out1 = tf.nn.relu(out)

    out = tf.layers.conv1d(out1,num_features,kernel_size,padding='SAME')
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out,rate=dropout_rate)
    out = tf.layers.conv1d(out,num_features,kernel_size,stride=2,padding='SAME')

    out1 = tf.layers.max_pooling1d(out1,kernel_size,2,'SAME')
    out = out + out1
    return out

def build_block(inpt, num_features, kernel_size, is_training, dropout_rate=0.5):
    out = tf.layers.batch_normalization(inpt, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out,dropout_rate)
    out = tf.layers.conv1d(out,num_features,kernel_size,padding='SAME')
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out,dropout_rate)
    out = tf.layers.conv1d(out,num_features,kernel_size,stride=2,padding='SAME')
    inpt = tf.layers.max_pooling1d(inpt,kernel_size,2,'SAME')
    out = out + inpt
    return out

def final_block(inpt,num_class, is_training):
    out = tf.layers.batch_normalization(inpt, is_training)
    out = tf.nn.relu(out)
    out = tf.layers.flatten(out)
    out1 = tf.layers.dense(out, num_class)
    out2 = tf.layers.dense(out, 1)
    end_points = {'class_end':out1,'speed_end': out2}
    return end_points

def cardinet(inpt,inpt_size,is_training):
    num_features = 32
    kernel_size = 16
    num_build_blocks = 8
    feature_increase_each_n_block = 2
    print('input shape',inpt.get_shape().as_list())
    # inpt = tf.expand_dims(tf.expand_dims(inpt,-1),-1)
    inpt = tf.reshape(inpt, [-1, inpt_size, 1])

    out = first_block(inpt, num_features, kernel_size, is_training, dropout_rate=0.8)

    for ii in range(num_build_blocks):
        if not ii % feature_increase_each_n_block:
            num_features *= 2
        out = build_block(out, num_features, kernel_size, is_training, dropout_rate=1)
    
    out = final_block(out, 3, is_training)

    return out



    