#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:24:08 2018

@author: codeplay2018
"""
import numpy as np


class DataInfo():
    def __init__(self, datasets, params):
        self.params = params
        self.datasets = datasets
    def get_params(self):
        return self.params

class ImgDataSet():
    # arguments:
    # imagelist, labelist, the first dimension must be number of examples
    # these two lists must contain elements of numpy array type
    # if shuffle == True, all the data will be arranged randomly
    def __init__(self, image_array=np.array([]),
                 label_array=np.array([]), speed_array=np.array([]), 
                 shuffle=False, use_speed=False):
        self.clean()
        self.images = np.array(self._imagelist)
        self.labels = np.array(self._labellist)
        self.speeds = np.array(self._speedlist)
        self.use_speed = use_speed
        if image_array.size:
            self.images = image_array[np.arange(image_array.shape[0])]
            self.labels = label_array[np.arange(label_array.shape[0])]
            self._num_examples = self.images.shape[0]
            if use_speed:
                self.speeds = speed_array[np.arange(speed_array.shape[0])]
        if shuffle:
            self.shuffle()
        self._index_in_epoch = 0

    def add_data(self, img, label, speed=None): 
        self._imagelist.append(list(img))
        self._labellist.append(list(label))
        if self.use_speed:
            self._speedlist.append(list(speed))
    
    # generally, when packaging data, when join the datasets, set clean=False
    # otherwise, set clean=True    
    def make(self, shuffle=True, clean=False):
        if len(self._imagelist):
            self.images = np.array(self._imagelist)
            self.labels = np.array(self._labellist)
            if self.use_speed:
                self.speeds = np.array(self._speedlist)
        else:
            self.add_data(self.images, self.labels, self.speeds)
        if shuffle:
            self.shuffle()
        if clean:
            self.clean()
        else:
            self._imagelist = list(self.images)
            self._labellist = list(self.labels)
            if self.use_speed:
                self._speedlist = list(self.speeds)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError('number of examples is not compatible with labels')
        self._num_examples = self.num_examples()

    def clean(self):
        self._imagelist = []
        self._labellist = []
        self._speedlist = []

    def shuffle(self):
        index = list(range(self.num_examples()))
        np.random.shuffle(index)
        self.images = self.images[index] # randomly arranged
        self.labels = self.labels[index] # randomly arranged
        if self.use_speed:
            self.speeds = self.speeds[index] # randomly arranged

    def num_examples(self):
        return self.images.shape[0] # return the first dimension, num of samples
    
    def next_batch(self, batchsize, shuffle=False,step_forward=True):
        if self._index_in_epoch + batchsize >= self._num_examples:
            self._index_in_epoch = 0
            shuffle = True
            is_epoch_over = True
        else:
            is_epoch_over = False
        start = self._index_in_epoch
        end = start + batchsize
        if shuffle:
            self.shuffle()
        if step_forward:
            self._index_in_epoch += batchsize
        batch = [self.images[range(start, end)], self.labels[range(start, end)]]
        if self.use_speed:
            batch.append(self.speeds[range(start, end)])
        return batch, is_epoch_over
    
    def seperate_data(self, sep=[0.5,0.5], verbose=False, shuffle=True):
        sep = list(sep)
        num = self.num_examples()
        subset1_end = int(num * sep[0])
        _tempImages = self.images
        _tempLabels = self.labels
        _tempSpeeds = self.speeds
        self.subset1 = ImgDataSet(_tempImages[:subset1_end], _tempLabels[:subset1_end],
                                  _tempSpeeds[:subset1_end], use_speed=self.use_speed)
        self.subset2 = ImgDataSet(_tempImages[subset1_end:], _tempLabels[subset1_end:],
                                  _tempSpeeds[subset1_end:], use_speed=self.use_speed)
        del _tempImages
        del _tempLabels
        del _tempSpeeds
        self.subset1.make(shuffle=shuffle,clean=True)
        self.subset2.make(shuffle=shuffle,clean=True)
        if verbose:
            print("num of seperated subset1 is %d, num of subset2 validset is %d"%(
                    self.subset1.num_examples(),self.subset2.num_examples()))
    
    def join_data(self, other):
        if self.num_examples() != 0:
            self.images = np.concatenate((self.images, other.images), axis=0)
            self.labels = np.concatenate((self.labels, other.labels), axis=0)
            if self.use_speed:
                self.speeds = np.concatenate((self.speeds, other.speeds), axis=0)
                
        else:
            self.images = other.images[np.arange(other.num_examples())]
            self.labels = other.labels[np.arange(other.num_examples())]
            if self.use_speed:
                self.speeds = other.speeds[np.arange(other.num_examples())]
    
    def isEmpty(self):
        return True if self.num_examples() == 0 else False