#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:27:51 2018

@author: codeplay2018
"""
import tensorflow as tf
from networks import net_factory, tf_utils

class cnn_model():
    def __init__(self,
                 netname, 
                 input_size,
                 num_class,
                 batch_size=None,
                 use_speed=False,
                 speed_loss_factor=0,
                 learning_rate=1e-4,
                 **kwargs):
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.class_label = tf.placeholder(tf.float32, [None, num_class])
        self.speed_label = tf.placeholder(tf.float32, [None,1])
        self.is_training = tf.placeholder(tf.bool)
        self.network = net_factory.get_network(netname)
        self.end_points = self.network(self.x, input_size, self.is_training)
        self.batch_size = batch_size
        self.use_speed = use_speed
        self.speed_loss_factor = speed_loss_factor
        self.learning_rate=learning_rate
        self._define_loss()
        self._define_evaluate()
        self._define_optimize()
        
    def get_feed(self, batch, is_training):
        if batch:
            feed_dict = {self.x: batch[0], self.class_label: batch[1],
                     self.speed_label: batch[2], self.is_training:is_training}
        else: feed_dict = None
        return feed_dict
        
    
    def _define_loss(self):
        
        with tf.name_scope('fault-loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.class_label,
                                                                logits=self.end_points['class_end'])
            self.cross_entropy_loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss-cross_entropy', self.cross_entropy_loss)
            tf.losses.add_loss(self.cross_entropy_loss)
        if self.use_speed:    
            with tf.name_scope('speed-loss'):
                speed_loss = tf_utils.abs_smooth(self.end_points['speed_end'] - self.speed_label)
                self.speed_loss = tf.div(tf.reduce_sum(self.speed_loss_factor*speed_loss),
                                    self.batch_size, name='value')
                tf.summary.scalar('loss-speed_loss', self.speed_loss)
                tf.losses.add_loss(self.speed_loss)
    
        with tf.name_scope('total_loss'):
            self.total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
    
    def _define_evaluate(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.end_points['class_end'], 1), 
                                          tf.argmax(self.class_label, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', self.accuracy)
    
    def _define_optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('adam_optimizer'):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
    