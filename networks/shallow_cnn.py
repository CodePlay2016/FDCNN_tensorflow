import tensorflow as tf
import networks.tf_utils as tu

def shallow2(inpt, inpt_size, is_training):
    '''
        18-ticnn
    '''
    end_points = {}
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first conv-----------------------------
    num_feature = 16
    out = tu.add_conv1d_layer(x_image, num_feature, 9, 2, is_training=is_training, 
                            layer_name='conv1')
    end_points['conv1']=out
    out = tu.batch_normalization(out, is_training=is_training)
    out = tu.max_pool(out, ksize=4, stride=2, layer_name='pool1')
    end_points['pool1']=out
    tu.print_activations(out) 
        
    out = tu.add_conv1d_layer(out, num_feature, 9, 2, is_training=is_training, 
                            layer_name='conv2')
    end_points['conv2']=out
    out = tu.max_pool(out, ksize=4, stride=2, layer_name='pool2')
    end_points['pool2']=out
    tu.print_activations(out) 
        
    ## fully connected layers----------------------------- 
    with tf.name_scope('fc1'):   
        out = tu.add_fc_layer(out, 100, relu=True, BN=True, 
                                is_training=is_training)# previously 256 nodes
        out = tf.nn.dropout(out, keep_prob)
    with tf.name_scope('fc2-1'):
        out1 = tu.add_fc_layer(out, 3)
        
    with tf.name_scope('fc2-2'):
        print(end_points.keys())
        out2 = tu.add_fc_layer(out, 1)

    end_points['class_end'] = out1
    end_points['speed_end'] = out2
    return end_points

def shallow1(inpt, inpt_size,
                is_training):
    '''
        18-ticnn
    '''
    end_points = {}
    keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
    with tf.name_scope('reshape'):
        x_image = tf.reshape(inpt, [-1, inpt_size,1, 1])

    ## first conv-----------------------------
    num_feature = 16
    out = tu.add_conv1d_layer(x_image, num_feature, 64, 8, BN=False, 
                            layer_name='conv1')
    end_points['conv1']=out
    out = tu.batch_normalization(out, is_training=is_training)
    out = tu.max_pool(out, ksize=2, stride=2, layer_name='pool1')
    end_points['pool1']=out
    tu.print_activations(out) 
        
    ## second conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 3, is_training=is_training, 
                            layer_name='conv2', print_activation=True)
    end_points['conv2']=out
    out = tu.max_pool(out, ksize=2,stride=2, layer_name='pool2')
    end_points['pool2']=out
    tu.print_activations(out)

    ## second conv-----------------------------
    num_feature = num_feature*2
    out = tu.add_conv1d_layer(out, num_feature, 3, is_training=is_training, 
                            layer_name='conv3', print_activation=True)
    end_points['conv3']=out
    out = tu.max_pool(out, ksize=2,stride=2, layer_name='pool3')
    end_points['pool3']=out
    tu.print_activations(out)

    out = tu.add_conv1d_layer(out, num_feature, 3, is_training=is_training, 
                            layer_name='conv3', print_activation=True)
    end_points['conv3']=out
    out = tu.max_pool(out, ksize=2,stride=2, layer_name='pool3')
    end_points['pool3']=out
    tu.print_activations(out)

    out = tu.add_conv1d_layer(out, num_feature, 3, is_training=is_training, 
                            layer_name='conv4', print_activation=True)
    end_points['conv4']=out
    out = tu.max_pool(out, ksize=2,stride=2, layer_name='pool4')
    end_points['pool4']=out
    tu.print_activations(out)

    out = tu.add_conv1d_layer(out, num_feature, 3, is_training=is_training, 
                            layer_name='conv5', print_activation=True)
    end_points['conv5']=out
    out = tu.max_pool(out, ksize=2,stride=2, layer_name='pool5')
    end_points['pool5']=out
    tu.print_activations(out)

    out = tu.add_conv1d_layer(out, num_feature, 3, is_training=is_training, 
                            layer_name='conv6', print_activation=True)
    end_points['conv6']=out
    out = tu.max_pool(out, ksize=2,stride=2, layer_name='pool6')
    end_points['pool6']=out
    tu.print_activations(out)
        
    ## fully connected layers----------------------------- 
    with tf.name_scope('fc1'):   
        out = tu.add_fc_layer(out, 100, relu=True, BN=True, 
                                is_training=is_training)# previously 256 nodes
        out = tf.nn.dropout(out, keep_prob)
    with tf.name_scope('fc2-1'):
        out1 = tu.add_fc_layer(out, 3)
        
    with tf.name_scope('fc2-2'):
        print(end_points.keys())
        out2 = tu.add_fc_layer(out, 1)

    end_points['class_end'] = out1
    end_points['speed_end'] = out2
    return end_points