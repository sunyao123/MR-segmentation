# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:02:04 2018

@author: sy
"""

import numpy as np

import tensorflow as tf
from tensorflow.python.training import moving_averages

regularizer_rate = 0.1

    
class segmentation_model():
    def __init__(self, scale_1_train, scale_2_train, scale_3_train, y_train, out_dim, dropout_rate):
        self.scale_1_train = scale_1_train
        self.scale_2_train = scale_2_train
        self.scale_3_train = scale_3_train
        self.y_train = y_train
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
    
    
    def _get_variable(self, name,
                  shape,
                  initializer,
                  regularizer= None,
                  dtype= 'float',
                  trainable= True):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name,
                               shape= shape,
                               initializer= initializer,
                               dtype= dtype,
                               regularizer = regularizer)
            # tf.summary.scalar(var.name+'/sparsity', tf.nn.zero_fraction(var))
            # tf.summary.histogram(var.name, var)
        return var
    
    def flatten(self, x):
        shape = x.get_shape().as_list()
        dim = 1
    #    for i in xrange(1,len(shape)):
        for i in range(1,len(shape)):
            dim*=shape[i]
        return tf.reshape(x, [-1, dim])
    
    def maxPool(self, x, ksize, stride):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
        
        
    def spatialConvolution(self, x, ksize, stride, filters_out, weight_initializer= None, bias_initializer= None):
        para = []
#        filters_in = x.get_shape()[-1]
        filters_in = x.shape[-1]
        stddev = 1./tf.sqrt(tf.cast(filters_out, tf.float32))
        if weight_initializer is None:
            weight_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)
        if bias_initializer is None:
            bias_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32) 
    
        shape = [ksize, ksize, filters_in, filters_out]
        weights = self._get_variable('weights',
                                shape, weight_initializer, regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate))
        
        conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding= 'VALID')
        biases = self._get_variable('biases', [filters_out],  bias_initializer, regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate))
        para.append(weights)
        para.append(biases)
        return tf.nn.bias_add(conv, biases)
    

    def fullyConnected(self, x, num_units_out, weight_initializer= None, bias_initializer= None):
        para = []
        num_units_in = x.get_shape()[1]
        stddev = 1./tf.sqrt(tf.cast(num_units_out, tf.float32))
        if weight_initializer is None:
            weight_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32)
        if bias_initializer is None:
            bias_initializer = tf.random_uniform_initializer(minval= -stddev, maxval= stddev, dtype= tf.float32) 
    
        weights = self._get_variable('weights',
                                [num_units_in, num_units_out], weight_initializer, regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate))
        biases = self._get_variable('biases',
                               [num_units_out], bias_initializer, regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate))
        para.append(weights)
        para.append(biases)
        return tf.nn.xw_plus_b(x, weights, biases)

   
        
    def batchNormalization(self, x, phase_train=True, scope='bn_conv'):
      with tf.variable_scope(scope):
          n_out = x.shape[-1]
          beta = tf.get_variable('beta_conv', shape=[n_out], initializer=tf.constant_initializer(0.0))
          gamma = tf.get_variable('gamma_conv', shape=[n_out], initializer=tf.constant_initializer(1.0))
    
          batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    
          moving_mean = tf.get_variable('batch_mean', shape=batch_mean.get_shape(), initializer=tf.constant_initializer(0.0), trainable=False)
          moving_variance = tf.get_variable('batch_var', shape=batch_var.get_shape(), initializer=tf.constant_initializer(0.0), trainable=False)
    
          update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                     batch_mean, 0.5, zero_debias=False)
          update_moving_variance = moving_averages.assign_moving_average(
                                                              moving_variance, batch_var, 0.5, zero_debias=False)
          def mean_var_with_update():
              with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                      return tf.identity(batch_mean), tf.identity(batch_var)
          mean, var = tf.cond(phase_train,
                                            mean_var_with_update,
                                            lambda: (moving_mean, moving_variance))
          normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
      return normed

    
    def scale_channel(self, kernal_s, kernal_out, name_first, last_maxpooling = False):
        with tf.variable_scope(name_first+'conv1'):
            if name_first.split('_')[1] == '1':
                network = self.spatialConvolution(self.scale_1_train, kernal_s[0], 1, kernal_out[0])
            elif name_first.split('_')[1] == '2':
                network = self.spatialConvolution(self.scale_2_train, kernal_s[0], 1, kernal_out[0])
            else:
                network = self.spatialConvolution(self.scale_3_train, kernal_s[0], 1, kernal_out[0])
            network = self.batchNormalization(network, self.phase_train)        
            network = tf.nn.relu(network)
        network = self.maxPool(network, 2, 2)
        with tf.variable_scope(name_first+'conv2'):
            network = self.spatialConvolution(network, kernal_s[1], 1, kernal_out[1])
            network = self.batchNormalization(network, self.phase_train)        
            network = tf.nn.relu(network)
        network = self.maxPool(network, 2, 2)        
        with tf.variable_scope(name_first+'conv3'):
            network = self.spatialConvolution(network, kernal_s[2], 1, kernal_out[2])
            network = self.batchNormalization(network, self.phase_train)        
            network = tf.nn.relu(network)
        if last_maxpooling:
            network = self.maxPool(network, 2, 2)
        return network
        
    def build_modul(self, train_phase):
        self.phase_train = train_phase
        channel_1_out = self.scale_channel(kernal_s=(5, 3, 3), kernal_out=(24, 32, 48), name_first='scale_1_', last_maxpooling=False)
        channel_1_out = self.flatten(channel_1_out)
        with tf.variable_scope('ful_1_out'):
            ful_1_out = self.fullyConnected(channel_1_out, 256)
        ful_1_out = tf.nn.dropout(ful_1_out, self.dropout_rate)
        fla_ful_1_out = tf.nn.relu(ful_1_out)
        
        channel_2_out = self.scale_channel(kernal_s=(7, 5, 3), kernal_out=(24, 32, 48), name_first='scale_2_', last_maxpooling=True)
        channel_2_out = self.flatten(channel_2_out)
        with tf.variable_scope('ful_2_out'):
            ful_2_out = self.fullyConnected(channel_2_out, 256)
        ful_2_out = tf.nn.dropout(ful_2_out, self.dropout_rate)
        fla_ful_2_out = tf.nn.relu(ful_2_out)
        
        channel_3_out = self.scale_channel(kernal_s=(9, 7, 5), kernal_out=(24, 32, 48), name_first='scale_3_', last_maxpooling=True)
        channel_3_out = self.flatten(channel_3_out)
        with tf.variable_scope('ful_3_out'):
            ful_3_out = self.fullyConnected(channel_3_out, 256)
        ful_3_out = tf.nn.dropout(ful_3_out, self.dropout_rate)
        fla_ful_3_out = tf.nn.relu(ful_3_out)
        
        ful_merge_out = tf.concat([fla_ful_1_out, fla_ful_2_out, fla_ful_3_out], 1)
        ful_merge_out = tf.nn.dropout(ful_merge_out, self.dropout_rate)
        final_out = self.fullyConnected(ful_merge_out, self.out_dim)
        
        with tf.name_scope('loss'), tf.device('/cpu:0'):
            loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_out, labels=self.y_train))
            # tf.summary.scalar('loss', loss)
            
        with tf.name_scope('accuracu'), tf.device('/cpu:0'):
            correct_pred = tf.equal(tf.arg_max(final_out, 1), tf.arg_max(self.y_train, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
            # tf.summary.scalar('accuracy', accuracy)
            
        return loss, accuracy








