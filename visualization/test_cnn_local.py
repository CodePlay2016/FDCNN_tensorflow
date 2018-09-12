#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:31:06 2017

@author: codeplay2017
"""


#####-----------structure parameters-----------------
INPUT_SIZE = 8192

####-----------------------------------------------------------------------
import time, os, sys, pickle, argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/home1/TestDataFromWen/arranged/steady_condition/pkl/')
    parser.add_argument("--model_path", type=str, default='checkpoint/shallcnn/2018-06-06_231243/model.ckpt')
    parser.add_argument("--cwd", type=str, default='/home/ad/code/lab/python/FDCNN_tensorflow/')
    args = parser.parse_args()
    return args
args = get_args()
os.chdir(args.cwd)

####------------------------------------------------------------------------
sys.path.append('../')
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from data import data_factory
from networks import model_factory

data_dir = args.data_dir # ubuntu
model_path = os.path.join(args.cwd, args.model_path)
####-----------------------------------------------------------------------
#reader = pywrap_tensorflow.NewCheckpointReader(model_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
## Print tensor name and values
#keylist = []
#for key in var_to_shape_map:
#    keylist.append(key)
#list.sort(keylist)
####-----------------------------------------------------------------------
####-----------------------------------------------------------------------

def main(): # _ means the last param
    # Create the model
    Model = model_factory.get_model('cnn')
    model = Model('cvgg19', INPUT_SIZE,3)
    
    ### load data
    test_speed = [50]
    data_fn = data_factory.get_data_from('wen')
    _, testset, _ = data_fn(data_dir,
                            samp_freq='12k',
                            speed_list=test_speed,
                            train_split=0.7,
                            vt_split=0,
                            divide_step=5,
                            data_length = INPUT_SIZE,
                            fft = False,
                            normalize=False,
                            verbose=False, use_speed=True)
    
    num_of_example = testset.num_examples()
    print('\nnumber of test examples is ', num_of_example)
    batch_size = 100
    
    plt.figure()
    out_dict = {}
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        with open("tsne_SCNN.pkl","rb") as f:
            out_dict = pickle.load(f)

        # test_batch,_ = testset.next_batch(batch_size)
        test_batch = out_dict["test_batch"]
        print(model.end_points.keys())
        test_feed = model.get_feed(test_batch,False)
        
        out = sess.run(model.end_points["pool2"], test_feed)
        out_dict["SCNN"] = out
        with open("tsne_SCNN.pkl","wb") as f:
            pickle.dump(out_dict,f)

    
    return test_batch


def plot_S_DCNN():
    with open("tsne_SCNN.pkl","rb") as f:
        out_dict = pickle.load(f)

    test_batch = out_dict["test_batch"]

    plt.figure(figsize=(10,5))
    ax = plt.subplot(1,2,1)
    ax = plot_tsne(out_dict["SCNN"], test_batch[1],ax)
    ax.set_title("SCNN-pool2")
    plt.ylim(-200,200)
    ax = plt.subplot(1,2,2)
    ax = plot_tsne(out_dict["DCNN"], test_batch[1],ax)
    ax.set_title("DCNN-GAP")
    plt.legend(["Normal","PG clipped tooth", "PG worn tooth"])
    plt.show()


def dofft(signal, fs):
    N = len(signal)
    fsignal = np.abs(np.fft.fft(signal))*2/N
    fsignal = fsignal[:N//2]
    f = fs*np.arange(0,N//2)/N
    return fsignal, f

def plot_tsne(data,labels,ax, **kwargs):
    print(data.shape)
    out_embedded = TSNE(n_components=2).fit_transform(data.reshape([data.shape[0],-1]))
    color_map = {
        0: 'red',
        1: 'green',
        2: 'blue'
    }
    for ii,point in enumerate(out_embedded):
        plt.scatter(point[0],point[1],c=color_map[np.argmax(labels[ii])], **kwargs)
    return ax
            
#########------custom functions----------------------------------
if __name__ == '__main__':
    plot_S_DCNN()
    # main()
