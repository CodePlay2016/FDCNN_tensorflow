#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:59:20 2018

@author: codeplay2018
"""

import tensorflow as tf

class svm_model():
    def __init__(self,
                 netname, 
                 input_size,
                 num_class,
                 batch_size,
                 speed_loss_factor=0,
                 learning_rate=1e-4,
                 **kwargs):
        self.num_class = num_class
        self.input_size = input_size
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.class_label = tf.placeholder(tf.float32, [None, num_class])
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.speed_loss_factor = speed_loss_factor
        self._forward()
        self._define_optimize()
        
    def get_feed(self, batch, is_training):
        if batch:
            feed_dict = {self.x: batch[0], self.class_label: batch[1]}
        else: feed_dict = None
        return feed_dict
    
    def _forward(self):
        with tf.name_scope('svm_model'):
            inpt = self.x
            target = tf.transpose(self.class_label)
            # Create variables for svm
            b = tf.Variable(tf.random_normal(shape=[self.num_class, self.batch_size]))
            
            # Gaussian (RBF) kernelnp.float32)
            gamma = tf.constant(-10.0)
            dist = tf.reduce_sum(tf.square(inpt), 1)
            dist = tf.reshape(dist, [-1, 1])
            sq_dists = tf.multiply(2., tf.matmul(inpt, tf.transpose(inpt)))
            my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
            
            # Declare function to do reshape/batch multiplication
            def reshape_matmul(mat, _size):
                v1 = tf.expand_dims(mat, 1)
                v2 = tf.reshape(v1, [self.num_class, _size, 1])
                return tf.matmul(v2, v1)    
            
            # Compute SVM Model
            first_term = tf.reduce_sum(b)
            b_vec_cross = tf.matmul(tf.transpose(b), b)
            y_target_cross = reshape_matmul(target, self.batch_size)
            
            second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
        
        with tf.name_scope('prediction'):
            # Gaussian (RBF) prediction kernel
            rA = tf.reshape(tf.reduce_sum(tf.square(inpt), 1), [-1, 1])
            rB = tf.reshape(tf.reduce_sum(tf.square(inpt), 1), [-1, 1])
            pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(inpt, tf.transpose(inpt)))), tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
        
            prediction_output = tf.matmul(tf.multiply(target, b), pred_kernel)
            self.prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(target, 0)), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        
        with tf.name_scope('loss_define'):
            self.loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
            tf.summary.scalar('loss', self.loss)
            tf.losses.add_loss(self.loss)
        
    
    def _define_optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('adam_optimizer'):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)