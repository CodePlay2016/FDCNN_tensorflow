#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:47:10 2017

@author: codeplay2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from builtins import str
## etc., as needed
#
##from future import standard_library
#standard_library.install_aliases()
import pickle, time, os, shutil, logging
import tensorflow as tf
import numpy as np
from data import data_factory
from networks import model_factory

# =========================================================================== #
# aliyun PAI flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'data_dir', '/home1/data/TestDataFromWen/arranged/steady_condition/pkl/',
    'dataset directory')
tf.app.flags.DEFINE_string(
    'checkpointDir', None, 'checkpoint saving directory')

# =========================================================================== #
# data flags.
# =========================================================================== #
tf.app.flags.DEFINE_string('dataset', 'wen', 'name of dataset, could be\n1.wen\n2.CWSU')
tf.app.flags.DEFINE_integer('input_size', 8192, 'length of input data')
tf.app.flags.DEFINE_integer('divide_step', 10, 'the step to divide the data')
tf.app.flags.DEFINE_string('train_speed_list', '40',
                           'a list of integer seperated by [,] which indicates '\
                           'under which speeds we train')
tf.app.flags.DEFINE_string('test_speed_list', '50',
                           'a list of integer seperated by [,] which indicates '\
                           'under which speeds we test(unknown set)')
tf.app.flags.DEFINE_bool('use_speed', True, 'whether to use speed to train')
tf.app.flags.DEFINE_integer('num_class', 3, 'number of classes')
tf.app.flags.DEFINE_float('train_split', 0.95,
                          'ratio to split the test data into valid and test,'\
                          'default is 0.5')
tf.app.flags.DEFINE_float('vt_split', 0.5,
                          'ratio to split the test data into valid and test,'\
                          'default is 0.5')
tf.app.flags.DEFINE_bool('do_norm', False, 'define whether to do normalization '\
                         'to the input data')
tf.app.flags.DEFINE_bool('do_fft', False, 'define whether to do fft to the input data')

# =========================================================================== #
# network flags.
# =========================================================================== #
tf.app.flags.DEFINE_string('network', 'cvgg19', 
                           'define which network to import, default is vgg19')
tf.app.flags.DEFINE_string('model_name', 'cnn', 
                           'define which model to import, default is cnn')

# =========================================================================== #
# training flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer('num_epoch', 10000, 'define the number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 50, 'define the batch size of each epoch')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'define the learning rate')
tf.app.flags.DEFINE_float('speed_loss_factor', 1.0, 'factor multiplied to the speed loss')

tf.app.flags.DEFINE_bool('early_stop', False, 'decide if need an early stop')
tf.app.flags.DEFINE_string('stop_standard', 'test',
                         'define which dataset we select as an early stop standard')
tf.app.flags.DEFINE_float('accuracy_threshold', 0.9,
                         'define accuracy maximum threshold during early stop')
tf.app.flags.DEFINE_float('accuracy_delta', 0.005,
                         'define accuracy maximum threshold during early stop')
tf.app.flags.DEFINE_float('loss_threshold', 0,
                         'define loss minimum threshold during early stop')


tf.app.flags.DEFINE_float('bn_epsilon', 1e-3, 'define the epsilon param in BN')
tf.app.flags.DEFINE_float('bn_momentum', 0.9, 'define the momentum param in BN')


FLAGS = tf.app.flags.FLAGS

#%%      
def main(_): # _ means the last param
    train_speed = list(map(int, FLAGS.train_speed_list.split(',')))
    test_speed = list(map(int, FLAGS.test_speed_list.split(',')))
    print('training domain: ', train_speed)
    print('adaptive domain: ', test_speed)
  
    logging.info("constructing graph..")
  # Create the model
    Model = model_factory.get_model(FLAGS.model_name)
    model = Model(FLAGS.network, FLAGS.input_size,
                  FLAGS.num_class,FLAGS.batch_size,
                  use_speed=FLAGS.use_speed,
                  speed_loss_factor=FLAGS.speed_loss_factor,
                  learning_rate=FLAGS.learning_rate)
    logging.info('use speed:' + str(model.use_speed))
    logging.info(model.speed_loss)
    logging.info('using model: '+FLAGS.network)
    
    # Import data
    logging.info('importing data...')
    data_fn = data_factory.get_data_from(FLAGS.dataset)
    trainset, validset, testset = data_fn(FLAGS.data_dir,
                                         speed_list=train_speed,
                                         train_split=FLAGS.train_split,
                                         vt_split=FLAGS.vt_split,
                                         divide_step=FLAGS.divide_step,
                                         data_length=FLAGS.input_size,
                                         fft=FLAGS.do_fft,
                                         normalize=FLAGS.do_norm,
                                         verbose=True, use_speed=FLAGS.use_speed)
    adavalidset, adatestset, adatestset2 = data_fn(FLAGS.data_dir,
                                         speed_list=test_speed,
                                         train_split=FLAGS.train_split,
                                         vt_split=FLAGS.vt_split,
                                         divide_step=FLAGS.divide_step,
                                         data_length = FLAGS.input_size,
                                         fft = False,
                                         normalize=False,
                                         verbose=False, use_speed=FLAGS.use_speed)
    # adatestset.join_data(adatestset2)
    logging.info('adavalidset and adatestset %d, %d'%(adavalidset.num_examples(),
                                               adatestset.num_examples()))
    
    
    # preparing the working directory, summaries
    time_info = time.strftime('%Y-%m-%d_%H%M%S',time.localtime(time.time()))

    output_dir = FLAGS.checkpointDir + time_info + '/'
    model_path = os.path.join(output_dir, 'model.ckpt')
    log_path   = os.path.join(output_dir, 'train.log')
    summary_path = os.path.join(output_dir, 'summary/')
    for end_point, x in model.end_points.items():
        tf.summary.histogram('activations/' + end_point, x)
        tf.summary.scalar('sparsity/' + end_point,
                                        tf.nn.zero_fraction(x))
    merged_summary = tf.summary.merge_all()
    logging.basicConfig(filename=log_path,filemode='a',format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=logging.DEBUG)

    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(summary_path+'train/',
                                                     sess.graph)
        valid_summary_writer = tf.summary.FileWriter(summary_path+'valid/')
        sess.run(tf.global_variables_initializer())
    
        saver = tf.train.Saver()
    
        curve_list = [[],[],[],[]]
        loss_this = 10 # give a initial loss
        start_time = time.time()
#        raise ValueError('stop')
        valid_batchsize = min(validset.num_examples(), FLAGS.batch_size)
        adavalid_batchsize = min(adavalidset.num_examples(), FLAGS.batch_size)
        high_perform = 0
        high_index = 0
        for i in range(FLAGS.num_epoch+1):
            train_batch, _ = trainset.next_batch(FLAGS.batch_size)
            valid_batch, _ = validset.next_batch(valid_batchsize)
            adavalid_batch, _ = adavalidset.next_batch(adavalid_batchsize)
            
            train_train_feed = model.get_feed(train_batch,True)
            train_eval_feed = model.get_feed(train_batch,False)
            valid_feed      = model.get_feed(valid_batch,False)
            adavalid_feed    = model.get_feed(adavalid_batch,False)
            if FLAGS.stop_standard == 'valid':
                loss_feed = valid_feed
            elif FLAGS.stop_standard == 'train':
                loss_feed = train_eval_feed
            else:
                loss_feed = adavalid_feed
            
            # if FLAGS.stop_standard == 'adatest':
            #     acc_this = acc_this*0.2 + valid_accuracy * 0.8 
            # update the highest performance checkpoint

            # if valid_accuracy - 0.98 >= 0.005 and acc_this > high_perform:
            #     high_perform = acc_this
            #     high_index   = i
            #     saver.save(sess=sess, save_path=model_path)

            if i and i % 100 == 0: # and show
                train_accuracy, train_speed_loss = sess.run([model.accuracy, model.speed_loss],
                                                            feed_dict=train_eval_feed)
                adatest_accuracy, adatest_speed_loss = sess.run([model.accuracy, model.speed_loss],
                                                            feed_dict=adavalid_feed)
                loss_this,acc_this = sess.run([model.cross_entropy_loss,model.accuracy],feed_dict=loss_feed)
                valid_accuracy = model.accuracy.eval(feed_dict=valid_feed)
                
                train_summary = sess.run(merged_summary, feed_dict=train_eval_feed)
                valid_summary = sess.run(merged_summary, feed_dict=valid_feed)
                train_summary_writer.add_summary(train_summary, i)
                valid_summary_writer.add_summary(valid_summary, i)
       
                msg = ('step |%d|, train acc |%.2g|, valid |%.3g|,'+
                       'highest is |%.4g|, test |%.3g|.') % (
                        i, train_accuracy, valid_accuracy, high_perform, adatest_accuracy)
                msg += ' time cost %.2g s'% (time.time()-start_time)
                if FLAGS.speed_loss_factor:
                    msg += ' train sploss %.2g / ada sploss %.2g'% (train_speed_loss, adatest_speed_loss)
                start_time = time.time()
                logging.info(msg)
                print(msg)
        
                curve_list[0].append(train_accuracy)
                curve_list[1].append(valid_accuracy)
                curve_list[2].append(high_perform)
                curve_list[3].append(adatest_accuracy)
            if i and i % 1000 == 0:
                saver.save(sess=sess, save_path=model_path)
                logging.info('model regularly saved...')
            
            if FLAGS.early_stop and\
            valid_accuracy - 0.98 >= 0.005:
                if (np.abs(acc_this - FLAGS.accuracy_threshold) >= FLAGS.accuracy_delta and
                    loss_this >= FLAGS.loss_threshold):
                    model.train_step.run(feed_dict=train_train_feed)
                elif np.abs(acc_this - FLAGS.accuracy_threshold) < FLAGS.accuracy_delta:
                    print('accuracy satisfied for %s is %.3g..stop training' % (
                            FLAGS.stop_standard, acc_this))
                    saver.save(sess=sess, save_path=model_path)
                    break
                elif loss_this < FLAGS.loss_threshold:
                    print('loss satisfied for %s is %.3g..stop training' % (
                            FLAGS.stop_standard, loss_this))
                    saver.save(sess=sess, save_path=model_path)
                    break
            else:
                model.train_step.run(feed_dict=train_train_feed)
        
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,model_path)    
        # save the trained model
        test_batch, _ = testset.next_batch(min(testset.num_examples(),1000))
        adatest_batch, _ = adatestset.next_batch(min(adatestset.num_examples(),1000))
        test_accuracy = model.accuracy.eval(feed_dict=model.get_feed(test_batch,False))
        adatest_accuracy = model.accuracy.eval(feed_dict=model.get_feed(adatest_batch,False))
        
    print('From %d test accuracy for train domain is %.3g' % (high_index,test_accuracy))
    print('test accuracy for adaptive domain is %.3g' % adatest_accuracy)

    with tf.gfile.GFile(output_dir+'curvelist.pkl', 'wb') as f:
        pickle.dump(curve_list, f)
    print('model is saved to:'+model_path)
    print('max valid accuracy is %.3g' % max(curve_list[1]))

#%%
if __name__ == '__main__':
    print('begin ')
    tf.app.run(main=main)
