    # -*- coding: utf-8 -*-
import tensorflow as tf

#####-----------structure parameters-----------------

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

def add_conv1d_layer(inpt, num_features, kernel_size, stride=1,
                     is_training=False, print_activation=False,
                     BN=True,layer_name=None):
    with tf.name_scope(layer_name):
        num_feature_pre = inpt.get_shape().as_list()[3]
        num_feature_cur = num_features
        W_conv = weight_variable([kernel_size, 1, num_feature_pre, num_feature_cur],
                 name='W_conv')
        b_conv = bias_variable([num_feature_cur], name='b_conv')
        out = conv2d(inpt, W_conv, strides=[1, stride, 1, 1])+b_conv
        if BN:
            out = batch_normalization(out, is_training=is_training)
        out = tf.nn.relu(out, name='h_conv')
        if print_activation:
            print_activations(out)
    return out

def add_fc_layer(inpt, out_size,is_training=True, relu=False, BN=False):
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

def global_average_pool(inpt, layer_name='GAP'):
    with tf.name_scope(layer_name):
        inpt_shape = inpt.get_shape().as_list()
        length = inpt_shape[1]
        out = tf.nn.avg_pool(inpt,ksize=[1,length,1,1], strides=[1,length,1,1],
                            padding="SAME")
    return out

def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())

def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)
    # notice this "SAME" param makes the conved image size the same as the original

def max_pool(x, ksize=2, stride=2, padding='SAME',layer_name=None):
    with tf.name_scope(layer_name):    
        out = tf.nn.max_pool(x, ksize=[1,ksize,1,1],
                            strides=[1,stride,1,1], padding=padding)
    return out

def max_pool_with_argmax(net, ksize, stride=2,layer_name=None):
    '''
    重定义一个最大池化函数，返回最大池化结果以及每个最大值的位置(是个索引，形状和池化结果一致)
    args:
        net:输入数据 形状为[batch,in_height,in_width,in_channels]
        stride：步长，是一个int32类型，注意在最大池化操作中我们设置窗口大小和步长大小是一样的
    '''
    with tf.name_scope(layer_name):
        #使用mask保存每个最大值的位置 这个函数只支持GPU操作
        _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, ksize, 1, 1], strides=[1, stride, 1, 1],padding='SAME')
        #将反向传播的mask梯度计算停止
        mask = tf.stop_gradient(mask)
        net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    return net,mask

def un_max_pool(net,mask,stride,layer_name):
    '''
    https://www.cnblogs.com/zyly/p/8991412.html
    定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
    args:
        net:最大池化后的输出，形状为[batch, height, width, in_channels]
        mask：位置索引组数组，形状和net一样
        stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
    '''
    ksize = [1, stride, 1, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

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

