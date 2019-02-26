#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 23:26:01 2018

@author: codeplay2018
"""
import networks.cnn_model as cnn_model
import networks.svm_model as svm_model

model_map = {'cnn':cnn_model.cnn_model,
             'svm':svm_model.svm_model,
            }

def get_model(modelname):
    return model_map[modelname]