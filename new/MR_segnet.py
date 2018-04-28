#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:11:59 2017

@author: sy
"""




from keras.layers import Input
from keras.layers.core import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import merge
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import numpy as np
import os
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from  keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras


l1 = keras.regularizers.l1(0.001)
l2 = keras.regularizers.l2(0.001)
#keras.regularizers.l1_l2(0.)


out_dim = 4

def different_patch(input_patch, kernel_size, kernel_channel, is_last_pooling=True):
    conv1 = Conv2D(kernel_channel[0], (kernel_size[0], kernel_size[0]), padding='valid', kernel_regularizer=l2, bias_regularizer=l1)(input_patch)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2),  padding='same')(conv1)
    
    conv2 = Conv2D(kernel_channel[1], (kernel_size[1], kernel_size[1]), padding='valid', kernel_regularizer=l2, bias_regularizer=l1)(conv1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2),  padding='same')(conv2)    

    conv3 = Conv2D(kernel_channel[2], (kernel_size[2], kernel_size[2]), padding='valid', kernel_regularizer=l2, bias_regularizer=l1)(conv2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    if is_last_pooling:
        conv3 = MaxPooling2D(pool_size=(2, 2),  padding='same')(conv3)
    flatten = Flatten()(conv3)
#    out = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(flatten)
    linear = Dense(256, activation='relu', kernel_regularizer=l2, bias_regularizer=l1)(flatten) 
#    linear = Dropout(rate=0.4)(linear)
    return linear

    
def get_unet():
    inputs_75 = Input((75, 75, 1))
    inputs_51 = Input((51, 51, 1))
    inputs_25 = Input((25, 25, 1))
    patch_75 = different_patch(inputs_75, kernel_size=(9, 7, 5), kernel_channel=(24, 32, 48), is_last_pooling=True)
    patch_51 = different_patch(inputs_51, kernel_size=(7, 5, 3), kernel_channel=(24, 32, 48), is_last_pooling=True)
    patch_25 = different_patch(inputs_25, kernel_size=(5, 3, 3), kernel_channel=(24, 32, 48), is_last_pooling=False)
    merge_all = merge([patch_75, patch_51, patch_25], mode='concat', concat_axis=1)
    out = Dense(out_dim, kernel_regularizer=l2, bias_regularizer=l1)(merge_all)
#    out = Dense(out_dim)(merge_all)
    out = Activation('softmax')(out)
    model = Model(input=[inputs_75, inputs_51, inputs_25], output=out)
#    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.3, nesterov=False)
    sgd = SGD(lr=0.01)
#    adam = Adam(lr=1e-1)
    model.summary()
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef])
    return model, sgd

get_unet()









