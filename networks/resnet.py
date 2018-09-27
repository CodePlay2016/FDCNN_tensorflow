# -*- coding: utf-8 -*-
import tensorflow as tf
import networks.tf_utils as tu
conv = tu.add_conv1d_layer

def res_block(inpt, out_channel, name_scope=None,down_sample=False):
    with tf.name_scope(name_scope):
        inpt = tf.cond(down_sample, 
            lambda: tu.add_conv1d_layer(inpt, out_channel, 3, 2, BN=True, layer_name='conv0'),
            lambda: inpt)
        out = tu.add_conv1d_layer(inpt, out_channel, 3, 1, BN=True, layer_name='conv1')
        out = tu.add_conv1d_layer(out, out_channel,3, 1, BN=True, layer_name='conv2')
    return inpt + out
        
def sphere_net20(inpt, inpt_size, is_training):
    end_points = {}
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        inpt = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first res-----------------------------
    num_feature = 16
    out = res_block(inpt, num_feature, name_scope='block1', down_sample=True)
    end_points['block1']=out
    tu.print_activations(out) 
        
    ## second res-----------------------------
    num_feature *= 2
    out = res_block(out, num_feature, name_scope='block2', down_sample=True)
    end_points['block2']=out
    out = res_block(out, num_feature, name_scope='block3', down_sample=False)
    end_points['block3']=out
    tu.print_activations(out) 
    
    ## third res-----------------------------
    num_feature *= 2
    out = res_block(out, num_feature, name_scope='block4', down_sample=True)
    end_points['block4']=out
    out = res_block(out, num_feature, name_scope='block5', down_sample=False)
    end_points['block5']=out
    out = res_block(out, num_feature, name_scope='block6', down_sample=False)
    end_points['block6']=out
    out = res_block(out, num_feature, name_scope='block7', down_sample=False)
    end_points['block7']=out
    tu.print_activations(out)
    
    ## forth conv-----------------------------
    num_feature *= 2
    out = res_block(out, num_feature, name_scope='block8', down_sample=True)
    end_points['block8']=out
    tu.print_activations(out)
    
    ## fully connected layers----------------------------- 
    with tf.name_scope('fc1_1'):   
        out = tu.add_fc_layer(out, 256, relu=True, BN=True, 
                                is_training=is_training)
        out = tf.nn.dropout(out, keep_prob)
    with tf.name_scope('fc1_2'):
        out = tu.add_fc_layer(out, 3)
    end_points['class_end'] = out
        
    with tf.name_scope('fc2_1'):
        print(end_points.keys())
        out = tu.add_fc_layer(end_points['block8'], 256, relu=True, BN=True,
                                        is_training=is_training)
        out = tf.nn.dropout(out, keep_prob)
    with tf.name_scope('fc2_2'):
        out = tu.add_fc_layer(out, 1)
    end_points['speed_end'] = out

    return end_points




