#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session  
#config = tf.ConfigProto()  
#
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
#config.gpu_options.allow_growth = True  
#set_session(tf.Session(config=config))


import sys
import numpy as np
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import random
from keras.utils import plot_model
import glob
import math
from keras.models import load_model
import time
import MR_segnet
from keras.callbacks import TensorBoard
from fuzzywuzzy import fuzz
import nibabel as nb
from glob import glob
import keras.backend as K
import tensorflow as tf
import shutil
import datetime
from keras.backend.tensorflow_backend import set_session
import random
import skimage.morphology as sm
import matplotlib.pyplot as plt
from keras.models import model_from_json

#==============================================================================
# 关于label
# 0：表示背景
# 1：脑脊液   CSF
# 2：灰质     JM
# 3：白质     WM
#==============================================================================


# 只在mask内提取训练集
# 为了防止mask内背景像素较少，所以对mask进行膨胀操作，这样可以处理边缘问题
# 膨胀像素点个数
dilation_value = 3
# 对每种大小的patch进行z-score

#统计的脊液：灰质：白质平均比例
proportion = (0.015, 0.64, 0.345)
#表示是否平衡类别，否则按照统计的比例来得到每类的数据量
class_balance = False

out_dim = 4
epochs = 100
every_class_number = None
#不同分辨率的输入大小
input_size = (75, 51, 25)


padding_size = 37
BATCH_SIZE = 64


path_of_data = '/home/ubuntu/MR-segmentation/data/Training_Set'
path_of_val = '/home/ubuntu/MR-segmentation/data/Validation_Set/'
path_of_weights = '/home/ubuntu/MR-segmentation/results/weights'
path_of_log = '/home/ubuntu/MR-segmentation/results/log'


imgs_all = None
label_all = None
mask_all = None
imgs_val = None
label_val = None
# 如果连续俩个周期交叉验证没有减少，那么减少学习率
patient = 4
# 减少学习为上一个周期学习率的drop_rate倍
drop_rate = 0.8



patient_to_stay = 0
# rejust_lr为true时可以调整学习率
rejust_lr = True
best_val_acc = 0

normalization = False

def generate_number():
    nowTime=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    randomNum=random.randint(0,1000)
    if int(randomNum) < 10:
        randomNum=str(00)+str(randomNum)
    if 10 <= int(randomNum) < 100:
        randomNum=str(0)+str(randomNum)
    uniqueNum=str(nowTime)+str(randomNum)
    return uniqueNum



def Mirroring(data):
    new_data = np.zeros((data.shape[0]+2*padding_size, data.shape[1]+2*padding_size, data.shape[2]))
    new_data[padding_size:new_data.shape[0]-padding_size, padding_size:new_data.shape[1]-padding_size, :] = data.reshape(data.shape[0], data.shape[1], data.shape[2])
    return new_data

def generate_train_data(phase='train'):
    global imgs_all, label_all, every_class_number, mask_all, class_balance
    if class_balance:
        every_class_number_0 = every_class_number
        every_class_number_1 = every_class_number
        every_class_number_2 = every_class_number
        every_class_number_3 = every_class_number
    else:
        every_class_number_0 = every_class_number*4*proportion[0]
        every_class_number_1 = every_class_number*4*proportion[0]
        every_class_number_2 = every_class_number*4*proportion[1]
        every_class_number_3 = every_class_number*4*proportion[2]
        
        
    if phase=='val':
        imgs_all = imgs_val
        label_all = label_val
    train_one_imgs = []
    train_one_label = []
    class_0_numbe = 0
    class_1_numbe = 0
    class_2_numbe = 0
    class_3_numbe = 0
    class_0_pointer = True
    class_1_pointer = True
    class_2_pointer = True
    class_3_pointer = True
    imgs_all = imgs_all.astype(np.float32)
# 生成mask内所有点的坐标
    index_to_extract_train_data = np.argwhere(mask_all != 0)
    while class_0_pointer or class_1_pointer or class_2_pointer or class_3_pointer:
        # z_random = random.randint(0, imgs_all.shape[2]-1)
        # x_random = random.randint(padding_size, imgs_all.shape[0]-padding_size-1)
        # y_random = random.randint(padding_size, imgs_all.shape[1]-padding_size-1)
        index_choosen = random.randint(0, len(index_to_extract_train_data)-1)
        x_random, y_random, z_random = index_to_extract_train_data[index_choosen]

        if label_all[x_random, y_random, z_random] == 0 and class_0_numbe<every_class_number_0:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            start_2 = int((input_size[0] - input_size[2]) / 2)
            imgs_tmp = imgs_one[start_2:start_2 + input_size[2], start_2:start_2 + input_size[2]]
            if imgs_tmp.max() == 0 or np.std(imgs_tmp) == 0:
                continue
            # imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)
            label_one = np.zeros(4)
            label_one[0] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_0_numbe += 1
        if class_0_numbe == every_class_number_0:
            class_0_pointer = False


        if label_all[x_random, y_random, z_random] == 1 and class_1_numbe<every_class_number_1:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            start_2 = int((input_size[0] - input_size[2]) / 2)
            imgs_tmp = imgs_one[start_2:start_2 + input_size[2], start_2:start_2 + input_size[2]]
            if imgs_tmp.max() == 0 or np.std(imgs_tmp)==0:
                continue
            # imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)
            label_one = np.zeros(4)
            label_one[1] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_1_numbe += 1
        if class_1_numbe == every_class_number_1:
            class_1_pointer = False


        if label_all[x_random, y_random, z_random] == 2 and class_2_numbe<every_class_number_2:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            start_2 = int((input_size[0] - input_size[2]) / 2)
            imgs_tmp = imgs_one[start_2:start_2 + input_size[2], start_2:start_2 + input_size[2]]
            if imgs_tmp.max() == 0 or np.std(imgs_tmp)==0:
                continue
            # imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)
            label_one = np.zeros(4)
            label_one[2] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_2_numbe += 1
        if class_2_numbe == every_class_number_2:
            class_2_pointer = False

        if label_all[x_random, y_random, z_random] == 3 and class_3_numbe<every_class_number_3:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            start_2 = int((input_size[0] - input_size[2]) / 2)
            imgs_tmp = imgs_one[start_2:start_2 + input_size[2], start_2:start_2 + input_size[2]]
            if imgs_tmp.max() == 0 or np.std(imgs_tmp)==0:
                continue
            # imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)
            label_one = np.zeros(4)
            label_one[3] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_3_numbe += 1
        if class_3_numbe == every_class_number_3:
            class_3_pointer = False

    train_one_imgs = np.array(train_one_imgs)
    train_one_imgs = train_one_imgs.reshape(train_one_imgs.shape[0], train_one_imgs.shape[1], train_one_imgs.shape[2], 1)
    train_one_label = np.array(train_one_label)
    return train_one_imgs, train_one_label


def load_data(phase='train'):
    global imgs_all, label_all, imgs_val, label_val, mask_all, normalization
    if phase == 'train':
        path_every = glob(os.path.join(path_of_data, '*'))
    elif phase == 'val':
        path_every = glob(os.path.join(path_of_val, '*'))
    pointer_read = True
    for i in path_every:
        for single_data in os.listdir(i):
            tmp_score = fuzz.partial_ratio("._", single_data)
            if tmp_score == 100:
                continue
            score_imgs = fuzz.partial_ratio("strip", single_data)
            score_label = fuzz.partial_ratio("segTRI_fill_ana", single_data)
            score_mask = fuzz.partial_ratio('ana_brainmask', single_data)
            if score_label == 100:
                label = nb.load(os.path.join(i, single_data)).get_data()
                label = label.astype(np.uint8)
#                label = tf.cast(label, tf.uint8)
                label = label.reshape(label.shape[0], label.shape[1], label.shape[2])
            elif score_imgs == 100:
                image = nb.load(os.path.join(i, single_data)).get_data()
                image = image.astype(np.float32)
#                image = tf.cast(image, tf.float32)
                image = image.reshape(image.shape[0], image.shape[1], image.shape[2])
                if normalization:
                    image = image*1023/np.max(image)
            elif score_mask == 100:
                mask = nb.load(os.path.join(i, single_data)).get_data()
                mask = mask.astype(np.uint8)
#                mask = tf.cast(mask, tf.uint8)
                mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2])
                # 对mask进行膨胀操作
                for mask_z in range(mask.shape[2]):
                    if mask[:,:,mask_z].max() > 0:
                        mask[:, :, mask_z] = sm.binary_dilation(mask[:,:,mask_z], selem=sm.disk(dilation_value))
        
        index_delete = []
        for single_z in range(image.shape[2]):
            if mask[:,:,single_z].max() <= 0:
                index_delete.append(single_z)
        image = np.delete(image, index_delete, axis=2)
        label = np.delete(label, index_delete, axis=2)
        mask = np.delete(mask, index_delete, axis=2)

        pad_image = Mirroring(image)
        pad_label = Mirroring(label)
        pad_mask = Mirroring(mask)
        if pointer_read:
            imgs_max = pad_image
            label_max = pad_label
            mask_max = pad_mask
            pointer_read = False
            continue
        imgs_max = np.concatenate((imgs_max, pad_image), axis=2)
        label_max = np.concatenate((label_max, pad_label), axis=2)
        mask_max = np.concatenate((mask_max, pad_label), axis=2)
        
    if phase == 'train':
        imgs_all = imgs_max
        label_all = label_max
        mask_all = mask_max
    elif phase == 'val':
        imgs_val = imgs_max
        label_val = label_max
        mask_all = mask_max

    # for zzz in range(imgs_all.shape[2]):
    #     plt.subplot(121)
    #     plt.imshow(imgs_all[:,:,zzz], cmap='gray')
    #     plt.subplot(122)
    #     plt.imshow(label_all[:,:,zzz], cmap='gray')
    #     plt.show()


def z_score(data_standardize, dim_to=-1):
    # data_standardize = np.squeeze(data_standardize)
    # mean_slices = np.mean(data_standardize, axis=(1,2))
    # std_slices = np.std(data_standardize, axis=(1,2))
    # # assert len(mean_slices)==data_standardize.shape[0]and data_standardize.shape[0]==len(std_slices)
    #
    # print(data_standardize.shape, mean_slices.shape, '$$$$$$$$$$$$$$$$$$$$$$')
    # dice = (data_standardize-mean_slices)/std_slices

    for axis_standardize in range(data_standardize.shape[0]):
        if np.std(data_standardize[axis_standardize]) != 0:
            data_standardize[axis_standardize] = (data_standardize[axis_standardize]- np.mean(data_standardize[axis_standardize]))/(np.std(data_standardize[axis_standardize]))
        else:
            data_standardize[axis_standardize] = (data_standardize[axis_standardize]- np.mean(data_standardize[axis_standardize]))/(np.std(data_standardize[axis_standardize])+0.001)
    return data_standardize



def checkout_data(imgs_train, imgs_label):
#首先检查每类数据是否类别平衡，如果不平衡报错，如果平衡，进行随机打乱。
#因为提取数据时，白质较多，可能集中被提取，所以数据较为集中。
    label_one_epoch = np.argmax(imgs_label, axis=-1)
    class_0_number = np.sum(label_one_epoch==0)
    class_1_number = np.sum(label_one_epoch==1)
    class_2_number = np.sum(label_one_epoch==2)
    class_3_number = np.sum(label_one_epoch==3)
    assert class_0_number==class_1_number
    assert class_1_number==class_2_number    
    assert class_2_number==class_3_number    
    
    list_all_data = list(range(len(imgs_label)))
    random.shuffle(list_all_data)
    new_imgs = [imgs_train[new_index_imgs] for new_index_imgs in list_all_data]
    new_imgs = np.array(new_imgs)
    new_label = [imgs_label[new_index_label] for new_index_label in list_all_data]
    new_label = np.array(new_label)
    assert new_imgs.shape==imgs_train.shape
    assert new_label.shape==imgs_label.shape
    return new_imgs, new_label
    
    
    
if __name__ == '__main__':

    # 指定GPU

#     固定显存的GPU
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth=True
#    config.gpu_options.per_process_gpu_memory_fraction = 0.3
#    set_session(tf.Session(config=config))

#     GPU按需使用GPU按需使用
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
#    session = tf.Session(config=config)
#    set_session(tf.Session(config=config))

    time_save = generate_number()

    # if os.path.exists(os.path.join(path_of_log, time_save, 'train')):
    #     shutil.rmtree(os.path.join(path_of_log, time_save, 'train'))
    #
    # if os.path.exists(os.path.join(path_of_weights, time_save)):
    #     shutil.rmtree(os.path.join(path_of_weights, time_save))

    if not os.path.exists(os.path.join(path_of_log, time_save, 'train')):
        os.makedirs(os.path.join(path_of_log, time_save, 'train'))

    if not os.path.exists(os.path.join(path_of_weights, time_save)):
        os.makedirs(os.path.join(path_of_weights, time_save))

    segment_MR_net, adam_to_lr = MR_segnet.get_unet()
# 保存模型
    with open(os.path.join(path_of_weights, time_save, 'segment_model.json'), 'w') as files:
        files.write(segment_MR_net.to_json())

#    segment_MR_net.load_weights(os.path.join(path_of_weights, '20180424074119818', 'segment_weights.h5'))
    
    
    # csv_logger = CSVLogger(os.path.join(path_of_log, 'train_epochs_log.csv'), append=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, epsilon=0, patience=2, min_lr=0.000001)
    # checkpointer = ModelCheckpoint(os.path.join(path_of_weights, 'weights.hdf5'), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
#==============================================================================
#     生成验证数据
#==============================================================================
    normalization = True
    load_data('val')
    print('验证集shape', imgs_val.shape)
    every_class_number = 2000
    imgs_val, label_val = generate_train_data(phase='val')
    assert imgs_val.shape[0] == every_class_number*4 and label_val.shape[0] == every_class_number*4
    imgs_val, label_val = checkout_data(imgs_val, label_val)


    start_1_val = int((input_size[0] - input_size[1]) / 2)
    batch_x_1_val = imgs_val[:, start_1_val:start_1_val + input_size[1], start_1_val:start_1_val + input_size[1], :]
    assert (batch_x_1_val.shape[1], batch_x_1_val.shape[2]) == (input_size[1], input_size[1])

    start_2_val = int((input_size[0] - input_size[2]) / 2)
    batch_x_2_val = imgs_val[:, start_2_val:start_2_val + input_size[2], start_2_val:start_2_val + input_size[2], :]
    assert (batch_x_2_val.shape[1], batch_x_2_val.shape[2]) == (input_size[2], input_size[2])

    # 对验证数据进行z-score处理
#    imgs_val = z_score(imgs_val, dim_to=-1)
#    batch_x_1_val = z_score(batch_x_1_val, dim_to=-1)
#    batch_x_2_val = z_score(batch_x_2_val, dim_to=-1)

    load_data('train')
    print('训练集shape', imgs_all.shape)

    train_writer = tf.summary.FileWriter(os.path.join(path_of_log, time_save, 'train'))
    train_writer.add_graph(tf.get_default_graph())

    for single_epochs in range(epochs):
        print('周期--', single_epochs, '-------------------------')
        time_before = time.clock()
        every_class_number = 10000
        train_imgs, train_label = generate_train_data(phase='train')
        assert train_imgs.shape[0] == every_class_number*4 and train_label.shape[0] == every_class_number*4
        train_imgs, train_label = checkout_data(train_imgs, train_label)
        print('load data time...', time.clock() - time_before)

        total_train_num = train_imgs.shape[0]
        total_batch = total_train_num // BATCH_SIZE
        
        for batch_idx in range(total_batch):
            
            batch_x_0 = train_imgs[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE, :, :, :]
            batch_y = train_label[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE, :]
            assert (batch_x_0.shape[1], batch_x_0.shape[2]) == (input_size[0], input_size[0])
            
            start_1 = int((input_size[0]-input_size[1])/2)
            batch_x_1 = batch_x_0[:, start_1:start_1+input_size[1], start_1:start_1+input_size[1], :]
            assert (batch_x_1.shape[1], batch_x_1.shape[2]) == (input_size[1], input_size[1])
            
            start_2 = int((input_size[0]-input_size[2])/2)
            batch_x_2 = batch_x_0[:, start_2:start_2+input_size[2], start_2:start_2+input_size[2], :]
            assert (batch_x_2.shape[1], batch_x_2.shape[2]) == (input_size[2], input_size[2])

#            batch_x_0 = z_score(batch_x_0, dim_to=-1)
#            batch_x_1 = z_score(batch_x_1, dim_to=-1)
#            batch_x_2 = z_score(batch_x_2, dim_to=-1)
            
            train_info = segment_MR_net.train_on_batch([batch_x_0, batch_x_1, batch_x_2], batch_y)
            if train_info[1]==0:
                print(batch_x_0.shape[0])
                for zz in range(batch_x_0.shape[0]):
                    plt.subplot(131)
                    plt.imshow(batch_x_0[zz, :, :, 0], cmap='gray')
                    plt.subplot(132)
                    plt.imshow(batch_x_1[zz, :, :, 0], cmap='gray')                    
                    plt.subplot(133)
                    plt.imshow(batch_x_2[zz, :, :, 0], cmap='gray')
                    

            train_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_info[0]), ])
            train_acc_summary = tf.Summary(value=[tf.Summary.Value(tag="train_acc", simple_value=train_info[1]), ])
            train_writer.add_summary(train_loss_summary, single_epochs*total_batch+batch_idx)
            train_writer.add_summary(train_acc_summary, single_epochs*total_batch + batch_idx)

        val_matrix = segment_MR_net.evaluate([imgs_val, batch_x_1_val, batch_x_2_val], label_val, sample_weight=None)
        print('验证损失:{0} 精确度:{1} 学习率:{2}'.format(val_matrix[0], val_matrix[1], K.get_value(adam_to_lr.lr)))

        val_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=val_matrix[0]), ])
        val_acc_summary = tf.Summary(value=[tf.Summary.Value(tag="val_acc", simple_value=val_matrix[1]), ])
        lr_summary = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=K.get_value(adam_to_lr.lr)), ])
        train_writer.add_summary(val_loss_summary, single_epochs)
        train_writer.add_summary(val_acc_summary, single_epochs)
        train_writer.add_summary(lr_summary, single_epochs)


# 动态调整学习率,保存最好模型
        if best_val_acc >= val_matrix[1] and rejust_lr:
            patient_to_stay += 1
        elif best_val_acc < val_matrix[1]:
            best_val_acc = val_matrix[1]
            patient_to_stay = 0
            segment_MR_net.save_weights(os.path.join(path_of_weights, time_save, 'segment_weights.h5'))

        if patient_to_stay == patient and rejust_lr:
            K.set_value(adam_to_lr.lr, drop_rate * K.get_value(adam_to_lr.lr))
            patient_to_stay = 0
        if K.get_value(adam_to_lr.lr) <= 0.0001:
            rejust_lr = False









    
    
    
    
    
    
#    path_of_data = '/home/sy/qq/tmp/'
#    path_of_goal = '/home/sy/qq/tmp1/'
#    if not os.path.exists(path_of_goal):
#        os.makedirs(path_of_goal)
#
#    segment_MR_net = MR_segnet.get_unet()
##    plot_model(classify_net, to_file=os.path.join(path_of_goal, 'model.png'), show_shapes =True, show_layer_names=True)
#    csv_logger = CSVLogger(path_of_goal + 'train_epochs_log.csv', append=True)
#    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, epsilon=0, patience=2, min_lr=0.000001)

    
    
    
    
#    early_stopping =EarlyStopping(monitor='val_loss', patience=2)
#    History = classify_net.fit(imgs_all, label_all, batch_size=8, epochs=epochs, shuffle=True, verbose=1, validation_split=0.2, initial_epoch=0, callbacks=[checkpointer, csv_logger, reduce_lr])                 
    
    
#    with open(os.path.join(path_of_goal, 'classify.json'), 'w') as files:
#        files.write(classify_net.to_json())
    
#    validation_split=0.2
#    epochs = 60
#    batch_size = 8
#    train_batch_size = 8
#    load_data_batch_size = train_batch_size
#    x_train, y_train, x_valid, y_valid = generate_train_validation_data_path(path_of_data, validation_split)
    
    
#    steps_per_epoch = math.ceil(len(x_train)/train_batch_size)
    
#    val_batch_size = 2500
#    validation_steps = math.ceil(len(x_valid)/val_batch_size)
    
#    model_weights_path = '/media/sy/software/my_paper/multy_classify/multy_classify_headneck/2/model_all.h5'
#    classify_net = load_model(model_weights_path)
    
#    classify_net.fit_generator(generate_batch_data_random(x_train, y_train, load_data_batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=True, verbose=1, validation_data=generate_batch_data_random(x_valid, y_valid, val_batch_size), validation_steps=validation_steps, initial_epoch=20, callbacks=[checkpointer, csv_logger, reduce_lr])

#    print('load data...........')
#    start = time.clock()
#    x_train, y_train = generate_train_validation_data(path_of_data)
#    end = time.clock()
#    print ('load data done, total time is %s'%(end-start))
#    History = classify_net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=validation_split, initial_epoch=0, callbacks=[checkpointer, csv_logger, reduce_lr, TensorBoard('./logs')])                 
#    History = classify_net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=validation_split, initial_epoch=0, callbacks=[csv_logger, reduce_lr, TensorBoard('./logs')])
#    classify_net.save_weights(os.path.join(path_of_goal, 'classify.h5'))


























