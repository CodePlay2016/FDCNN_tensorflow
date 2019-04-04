#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:23:46 2018

@author: codeplay2018
"""
import numpy as np
import tensorflow as tf
import pickle, os, platform
from data.dataset import ImgDataSet, DataInfo

def _prepare_data(filepath,step, data_length, fft, mirror,channel=3, normalize=False):
    source_type = filepath.split('/')[-1].split('_')[0]
    switcher = {"na": [1,0,0],
                "pmt":   [0,1,0],
                "psf":   [0,0,1]}
    if source_type in switcher.keys():
        data_type = switcher.get(source_type)
    else:
        raise ValueError('unknown class name')
    
    with tf.gfile.GFile(filepath,'rb') as pf:
        if platform.python_version_tuple()[0] == '3':
            rawdata = pickle.load(pf, encoding='iso-8859-1')
        else:
            rawdata = pickle.load(pf)
    rawdata = rawdata[:,channel]
    matdata = []
    for ii in range(0,50000, step):
        if ii+data_length > rawdata.shape[0]:
            break
        matdata.append(rawdata[ii:ii+data_length])
    
    matdata = np.array(matdata)
    
    if fft:
        matdata = np.abs(np.fft.fft(matdata))[:, 1:matdata.shape[0]//2+1]*2/data_length
    
    num_of_data, data_length = matdata.shape
    if normalize:
        mmean = np.matmul(np.mean(matdata,1).reshape(num_of_data, 1),
                         np.ones([1,data_length]))
        mstd = np.matmul(np.std(matdata,1).reshape(num_of_data, 1),
                         np.ones([1,data_length]))
        matdata = (matdata-mmean)/mstd
    
    if mirror:
        matdata = np.concatenate(
                (matdata, matdata[:,list(range(data_length-1,-1,-1))]))
        num_of_data, length = matdata.shape
#    print("the shape of data is "+str(matdata.shape))
    
     
    return matdata, data_type, source_type, num_of_data, data_length


def get_dataset(filepath, targetpath=None, samp_freq='12k',
                speed_list = [10,20,30,40,50],
                train_split=0.95,
                vt_split=0.5,
                divide_step=2,
                data_length = 4096,
                fft = False,
                mirror=False,
                normalize=False,
                verbose=False,
                use_speed=False):
    filename = os.path.join(filepath,'*_'+samp_freq+'*.pkl')
    file_list = tf.gfile.Glob(filename)
    list.sort(file_list)
    trainset, validset, testset = (ImgDataSet(use_speed=use_speed),
                                   ImgDataSet(use_speed=use_speed), 
                                   ImgDataSet(use_speed=use_speed))
    
#    param_record_flag = False
    params = {}
    params['speed_list'] = speed_list
    params['train_split'] = train_split
    params['divide_step'] = divide_step
    
    for infile in file_list:
#        if not os.path.exists(infile):
#            raise ValueError('input path |%s| does not exist' % infile)
        filename = infile.split('/')[-1]
        speed = int(filename.split('_')[2].split('.')[0])
        if speed not in speed_list: continue
        (matdata, data_type, source_type,
         num_of_data, new_data_length) = _prepare_data(infile,divide_step,data_length,
                                 fft, mirror,normalize=normalize)
#        if not param_record_flag:
#            params['']
        #for train set
        print(filename)
        dataset = ImgDataSet(use_speed=use_speed)
        train_endpoint = int(num_of_data * train_split)
        dataset.images = matdata[:train_endpoint,:]
        print(dataset.images.shape)
        dataset.labels = np.array([data_type]*train_endpoint)
        dataset.speeds = np.array([speed/2]*train_endpoint).reshape(-1,1)
        trainset.join_data(dataset)
        trainset.make(shuffle=False, clean=True)
#        if verbose:
#            print('trainset of %s contains %d examples '%(source_type, 
#                                                          dataset.num_examples()))

        
        dataset = ImgDataSet(use_speed=use_speed)
        test_startpoint = train_endpoint + data_length//divide_step
        dataset.images = matdata[test_startpoint:,:]
        dataset.labels = np.array([data_type]*(num_of_data-test_startpoint))
        dataset.speeds = np.array([speed/2]*(num_of_data-train_endpoint)).reshape(-1,1)
        validset.join_data(dataset)
        validset.make(shuffle=False, clean=True)
#        if verbose:
#            print('validset of %s contains %d '%(source_type, 
#                                                          dataset.num_examples()))
        del matdata
    trainset.shuffle()
    validset.shuffle()
    if vt_split > 0:
        validset.seperate_data(sep=[vt_split, 1-vt_split], verbose=verbose)
        validset, testset = (validset.subset1, validset.subset2)
    else:
        testset = ImgDataSet()
    if verbose:
        print('number of training examples is %d,\nnumber of validate examples is %d\
              \nnumber of testing examples is %d' % (trainset.num_examples(),
              validset.num_examples(), testset.num_examples()))
    if targetpath is not None and os.path.exists(targetpath):
        datasets = DataInfo([trainset, validset, testset], params)
        with tf.gfile.GFile(os.path.join(targetpath,'datasets.pkl'), 'wb') as pf:
            pickle.dump(datasets, pf)
    return trainset, validset, testset
