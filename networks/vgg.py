# -*- coding: utf-8 -*-
import tensorflow as tf
import networks.tf_utils as tu

def cvgg19(inpt, inpt_size,
                is_training):
    
    end_points = {}
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first conv-----------------------------
    num_feature = 16
    out = tu.add_conv1d_layer(x_image, num_feature, 9, 2, BN=False, 
                            layer_name='conv1')
    end_points['conv1']=out
    out = tu.add_conv1d_layer(out, num_feature, 9, 2, BN=False, 
                            layer_name='conv2')
    end_points['conv2']=out
    out = tu.batch_normalization(out, is_training=is_training)
    out = tu.max_pool(out, ksize=4, layer_name='pool1')
    end_points['pool1']=out

    tu.print_activations(out) 
        
    ## second conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, 
                            layer_name='conv3')
    end_points['conv3']=out
    
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv4', print_activation=True)
    end_points['conv4']=out
    out = tu.max_pool(out, ksize=4, layer_name='pool2')
    end_points['pool2']=out
    tu.print_activations(out)
        
    ## third conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv6')
    end_points['conv6'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv7')
    end_points['conv7'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv8', print_activation=True)
    end_points['conv8'] = out
    out = tu.max_pool(out, ksize=4, layer_name='pool3')
    end_points['pool3'] = out
    tu.print_activations(out)
    
    ## forth conv-----------------------------
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv10')
    end_points['conv10'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv11')
    end_points['conv11'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv12', print_activation=True)
    end_points['conv12'] = out
    # out = tu.max_pool(out, ksize=4, layer_name='pool4')
    out = tu.global_average_pool(out)
    end_points['GAP'] = out
    tu.print_activations(out)
    
    ## fully connected layers----------------------------- 
    with tf.name_scope('fc1'):   
        out = tu.add_fc_layer(out, 256, relu=True, BN=True, 
                                is_training=is_training)
        out = tf.nn.dropout(out, keep_prob)
#    with tf.name_scope('fc1'):
#        out = tu.add_fc_layer(out, 256, relu=True, BN=True, is_training=is_training)
#    with tf.name_scope('dropout1'):
#        out = tf.nn.dropout(out, keep_prob)
    with tf.name_scope('fc4'):
        out = tu.add_fc_layer(out, 3)
        
    with tf.name_scope('block8'):
        print(end_points.keys())
        out2 = tu.add_fc_layer(end_points['GAP'], 256, relu=True, BN=True,
                                        is_training=is_training)
        out2 = tf.nn.dropout(out2, keep_prob)
    with tf.name_scope('block9'):
        out2 = tu.add_fc_layer(out2, 1)

    end_points['class_end'] = out
    end_points['speed_end'] = out2
    return end_points

def cvgg19_2(inpt, inpt_size,
                is_training):
    
    end_points = {}
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first conv-----------------------------
    num_feature = 16
    out = tu.add_conv1d_layer(x_image, num_feature, 9, 2, BN=False, 
                            layer_name='conv1')
    end_points['conv1']=out
    out = tu.add_conv1d_layer(out, num_feature, 9, 2, BN=False, 
                            layer_name='conv2')
    end_points['conv2']=out
    out = tu.batch_normalization(out, is_training=is_training)
    out = tu.max_pool(out, ksize=4, layer_name='pool1')
    end_points['pool1']=out

    tu.print_activations(out) 
        
    ## second conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, 
                            layer_name='conv3')
    end_points['conv3']=out
    
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv4', print_activation=True)
    end_points['conv4']=out
    out = tu.max_pool(out, ksize=4, layer_name='pool2')
    end_points['pool2']=out
    tu.print_activations(out)
        
    ## third conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv6')
    end_points['conv5'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv7')
    end_points['conv6'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv8', print_activation=True)
    end_points['conv7'] = out
    out = tu.max_pool(out, ksize=4, layer_name='pool3')
    end_points['pool3'] = out
    tu.print_activations(out)
    
    ## forth conv-----------------------------
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv10')
    end_points['conv8'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv11')
    end_points['conv9'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv12', print_activation=True)
    end_points['conv10'] = out
    # out = tu.max_pool(out, ksize=4, layer_name='pool4')
    out = tu.global_average_pool(out)
    end_points['GAP'] = out
    tu.print_activations(out)
    
    ## fully connected layers----------------------------- 
    with tf.name_scope('fc1'):   
        out = tu.add_fc_layer(out, 256, relu=True, BN=True, 
                                is_training=is_training)
        out = tf.nn.dropout(out, keep_prob)
    with tf.name_scope('fc2-1'):
        out1 = tu.add_fc_layer(out, 3)
        
    with tf.name_scope('fc2-2'):
        out2 = tu.add_fc_layer(out, 1)

    end_points['class_end'] = out1
    end_points['speed_end'] = out2
    return end_points

def cvgg19_3(inpt, inpt_size,
                is_training, num_class=3):
    
    end_points = {}
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first conv-----------------------------
    num_feature = 16
    out = tu.add_conv1d_layer(x_image, num_feature, 9, 2, BN=False, 
                            layer_name='conv1')
    end_points['conv1']=out
    out = tu.add_conv1d_layer(out, num_feature, 9, 2, BN=False, 
                            layer_name='conv2')
    end_points['conv2']=out
    out = tu.batch_normalization(out, is_training=is_training)
    out = tu.max_pool(out, ksize=4, layer_name='pool1')
    end_points['pool1']=out

    tu.print_activations(out) 
        
    ## second conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, 
                            layer_name='conv3')
    end_points['conv3']=out
    
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv4', print_activation=True)
    end_points['conv4']=out
    out = tu.max_pool(out, ksize=4, layer_name='pool2')
    end_points['pool2']=out
    tu.print_activations(out)
        
    ## third conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv6')
    end_points['conv5'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv7')
    end_points['conv6'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, is_training=is_training, 
                            layer_name='conv8', print_activation=True)
    end_points['conv7'] = out
    out = tu.max_pool(out, ksize=4, layer_name='pool3')
    end_points['pool3'] = out
    tu.print_activations(out)
    
    ## forth conv-----------------------------
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv10')
    end_points['conv8'] = out
    out = tu.add_conv1d_layer(out, num_feature, 9, BN=False, layer_name='conv11')
    end_points['conv9'] = out
    out1 = tu.add_conv1d_layer(out, num_class, 1, is_training=is_training, 
                            layer_name='conv10', print_activation=True)
    out2 = tu.add_conv1d_layer(out, 1, 1, is_training=is_training, 
                            layer_name='conv11', print_activation=True)
    end_points['conv10'] = out1
    # out = tu.max_pool(out, ksize=4, layer_name='pool4')
    out1 = tu.global_average_pool(out1)
    out2 = tu.global_average_pool(out2)
    end_points['GAP'] = out1
    tu.print_activations(out1)
    out1 = tf.reshape(out1,[-1,num_class])
    out2 = tf.reshape(out2,[-1,1])
    print("out1"+str(out1.get_shape().as_list()))
    print("out2"+str(out2.get_shape().as_list()))
    end_points['class_end'] = out1
    end_points['speed_end'] = out2
    return end_points
