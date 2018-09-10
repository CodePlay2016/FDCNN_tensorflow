# -*- coding: utf-8 -*-
import tensorflow as tf

def add_conv1d_layer(inpt,num_features, kernel_size, is_training, stride=1,
                     print_activation=True,
                     BN=True):
    num_feature_pre = inpt.get_shape().as_list()[3]
    num_feature_cur = num_features
    W_conv = weight_variable([kernel_size, 1, num_feature_pre, num_feature_cur], name='W_conv')
    b_conv = bias_variable([num_feature_cur], name='b_conv')
    out = tf.nn.relu(conv2d(inpt, W_conv, 
                             strides=[1, stride, 1, 1])+b_conv, name='h_conv')
    if BN:
        out = batch_normalization(out, is_training=is_training)
    if print_activation:
        print_activations(out)
    return out

def add_fc_layer(inpt,out_size,is_training=True, relu=False, BN=False):
    inpt_shape = inpt.get_shape().as_list()
    if len(inpt_shape) == 4:
        in_size = inpt_shape[1]*inpt_shape[3]
        inpt = tf.reshape(inpt, [-1, in_size])
    elif len(inpt_shape) == 2:
        in_size = inpt_shape[1]
    W_fc = weight_variable([in_size, out_size], 'W_fc')
    b_fc = bias_variable([out_size], 'b_fc')
    out = tf.nn.relu(tf.matmul(inpt, W_fc) + b_fc, 'h_fc')
    if BN:
        out = batch_normalization(out, is_training=is_training)
    return out

def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())

def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)
    # notice this "SAME" param makes the conved image size the same as the original

def max_pool(x, ksize=[1,2,1,1], strides=[1,2,2,1], padding='SAME'):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)

def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# if wanna change version of this function, see utils/batch_normalizations.py
def batch_normalization(inputs, is_training, epsilon = 0.001, momentum=0.9):
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=False,
        training = is_training)

