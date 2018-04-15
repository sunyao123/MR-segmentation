#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import glob
import os
import numpy as np
import random
#from segmentation_net import *
import segmentation_net
import time
import sys
from glob import glob
from fuzzywuzzy import fuzz
import nibabel as nb
import matplotlib.pyplot as plt

#==============================================================================
# 关于label
# 0：表示背景
# 1：脑脊液
# 2：灰质
# 3：白质
#==============================================================================

BATCH_SIZE = 32

log_device_placement = False

EPOCH_NUM = 100

DROP_RATE = 0.5
num_output = 4
# path_of_data = r'C:\sunyao_document\data\data_out\\'
path_of_data = r'/media/dengy/我的文件/sunyao/Training_Set/'
path_of_val = r'/media/dengy/我的文件/sunyao/Validation_Set/'
# path_of_weights = r'C:\sunyao_document\data\out\weights'
path_of_weights = r'/media/dengy/我的文件/sunyao/weights'
# path_of_log = r'C:\sunyao_document\data\out\log'
path_of_log = r'/media/dengy/我的文件/sunyao/log'

every_class_number = 10000
#监测一个变量,如果这个变量在patient周期内没有提高,则降低学习率
patient = 2
wd = 0.0001
#下降学习率的比例
lr_decay_rate = 0.7

#训练集中，验证集比例，将第一个周期的val_rate作为验证集
val_rate = 0.2

#不同分辨率的输入大小
input_size = (75, 51, 25)

#创建一个空类，底下把各种资料都存到类中
class PARAMETERS(object):
    pass

imgs_all = None
label_all = None
imgs_val = None
label_val = None


padding_size = 37

def Mirroring(data):
    new_data = np.zeros((data.shape[0]+2*padding_size, data.shape[1]+2*padding_size, data.shape[2]))
    new_data[padding_size:new_data.shape[0]-padding_size, padding_size:new_data.shape[1]-padding_size, :] = data.reshape(data.shape[0], data.shape[1], data.shape[2])
    return new_data

def generate_train_data():
    global imgs_all, label_all
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
    while class_0_pointer or class_1_pointer or class_2_pointer or class_3_pointer:
        z_random = random.randint(0, imgs_all.shape[2]-1)
        x_random = random.randint(padding_size, imgs_all.shape[0]-padding_size-1)
        y_random = random.randint(padding_size, imgs_all.shape[1]-padding_size-1)
        if label_all[x_random, y_random, z_random] == 0 and class_0_numbe<every_class_number:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]

            start_2 = int((input_size[0] - input_size[2]) / 2)
            imgs_tmp = imgs_one[start_2:start_2 + input_size[2], start_2:start_2 + input_size[2]]
            if imgs_tmp.max() == 0:
                continue

            # print(imgs_one.max(), imgs_one.min(), np.mean(imgs_one), np.std(imgs_one))
            imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)

            label_one = np.zeros(4)
            label_one[0] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_0_numbe += 1
        if class_0_numbe == every_class_number:
            class_0_pointer = False

        if label_all[x_random, y_random, z_random] == 1 and class_1_numbe<every_class_number:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            if imgs_one.max() == 0:
                continue
            imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)
            label_one = np.zeros(4)
            label_one[1] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_1_numbe += 1
        if class_1_numbe == every_class_number:
            class_1_pointer = False

        if label_all[x_random, y_random, z_random] == 2 and class_2_numbe<every_class_number:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            if imgs_one.max() == 0:
                continue
            imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)

            label_one = np.zeros(4)
            label_one[2] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_2_numbe += 1
        if class_2_numbe == every_class_number:
            class_2_pointer = False

        if label_all[x_random, y_random, z_random] == 3 and class_3_numbe<every_class_number:
            imgs_one = imgs_all[x_random-padding_size:x_random+padding_size+1, y_random-padding_size:y_random+padding_size+1, z_random]
            if imgs_one.max() == 0:
                continue
            imgs_one = (imgs_one-np.mean(imgs_one))/np.std(imgs_one)

            label_one = np.zeros(4)
            label_one[3] = 1
            train_one_imgs.append(imgs_one)
            train_one_label.append(label_one)
            class_3_numbe += 1
        if class_3_numbe == every_class_number:
            class_3_pointer = False

    train_one_imgs = np.array(train_one_imgs)
    train_one_imgs = train_one_imgs.reshape(train_one_imgs.shape[0], train_one_imgs.shape[1], train_one_imgs.shape[2], 1)
    train_one_label = np.array(train_one_label)
    return train_one_imgs, train_one_label


def load_data(phase='train'):
    global imgs_all, label_all, imgs_val, label_val
    if phase == 'train':
        path_every = glob(os.path.join(path_of_data, '*'))
    elif phase == 'val':
        path_every = glob(os.path.join(path_of_val, '*'))
    pointer_read = True
    for i in path_every:
        if not os.path.isdir(i):
            continue
        for single_data in os.listdir(i):
            score_imgs = fuzz.partial_ratio("strip", single_data)
            score_label = fuzz.partial_ratio("segTRI_ana", single_data)
            if score_label == 100:
                label = nb.load(os.path.join(i, single_data)).get_data()
                label = label.reshape(label.shape[0], label.shape[1], label.shape[2])
            elif score_imgs == 100:
                image = nb.load(os.path.join(i, single_data)).get_data()
                image = image.reshape(image.shape[0], image.shape[1], image.shape[2])

        index_delete = []
        for single_z in range(image.shape[2]):
            if image[:,:,single_z].max() <= 0:
                index_delete.append(single_z)
        image = np.delete(image, index_delete, axis=2)
        label = np.delete(label, index_delete, axis=2)
        pad_image = Mirroring(image)
        pad_label = Mirroring(label)
        if pointer_read:
            imgs_max = pad_image
            label_max = pad_label
            pointer_read = False
        imgs_max = np.concatenate((imgs_max, pad_image), axis=2)
        label_max = np.concatenate((label_max, pad_label), axis=2)
        if phase == 'train':
            imgs_all = imgs_max
            label_all = label_max
        elif phase == 'val':
            imgs_val = imgs_max
            label_val = label_max
    # for zzz in range(imgs_all.shape[2]):
    #     plt.subplot(121)
    #     plt.imshow(imgs_all[:,:,zzz], cmap='gray')
    #     plt.subplot(122)
    #     plt.imshow(label_all[:,:,zzz], cmap='gray')
    #     plt.show()


def single_gpu():
    load_data('train')
    load_data('val')
# 提取验证集数据
    imgs_val, label_val = generate_train_data()

    start_1_val = int((input_size[0] - input_size[1]) / 2)
    batch_x_1_val = imgs_val[:, start_1_val:start_1_val + input_size[1], start_1_val:start_1_val + input_size[1], :]
    assert (batch_x_1_val.shape[1], batch_x_1_val.shape[2]) == (input_size[1], input_size[1])

    start_2_val = int((input_size[0] - input_size[2]) / 2)
    batch_x_2_val = imgs_val[:, start_2_val:start_2_val + input_size[2], start_2_val:start_2_val + input_size[2], :]
    assert (batch_x_2_val.shape[1], batch_x_2_val.shape[2]) == (input_size[2], input_size[2])

# =============================================================================
# 初始化设置
# =============================================================================
    tf.reset_default_graph()
    # 选择是否显示每个op和varible的物理位置
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    # 让gpu模式为随取随用而不是直接全部占满
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        images_0 = tf.placeholder(tf.float32, [None, 75, 75, 1])
        images_1 = tf.placeholder(tf.float32, [None, 51, 51, 1])
        images_2 = tf.placeholder(tf.float32, [None, 25, 25, 1])
        labels_one = tf.placeholder(tf.float32, [None, num_output])
        train_phase = tf.placeholder(tf.bool)
        
        model = segmentation_net.segmentation_model(images_2, images_1, images_0, labels_one, num_output, DROP_RATE)
        
        with tf.variable_scope(tf.get_variable_scope()):
#            with tf.device('/gpu:0'):
            with tf.device('/cpu:0'):
                loss, acc = model.build_modul(train_phase)
                grads = opt.compute_gradients(loss)
        aver_loss_op = loss
        apply_gradient_op = opt.apply_gradients(grads)
        aver_acc_op = acc
        
        loss_visual = tf.summary.scalar('loss', loss)
        acc_visual = tf.summary.scalar('loss', acc)

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
# =============================================================================
# 建立可视化
# =============================================================================
            writer_train = tf.summary.FileWriter(os.path.join(path_of_log, 'train'))
            writer_val = tf.summary.FileWriter(os.path.join(path_of_log, 'test'))
            
            writer_train.add_graph(sess.graph)
            # merged_all = tf.summary.merge_all()
            # 初始化学习率
            lr = 0.01
            _val_acc_max = 0
            _val_patient_pointer = 0
            for epoch in range(EPOCH_NUM):
# =============================================================================
# 导入数据
# =============================================================================
                time_before = time.clock()
                train_imgs, train_label = generate_train_data()
                print('load data time...', time.clock() - time_before)

                avg_loss = 0.0
                avg_acc = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.6f' % (epoch, lr))

                total_train_num = train_imgs.shape[0]
                total_batch = total_train_num // BATCH_SIZE
# =============================================================================
# 训练部分
# =============================================================================
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

                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict[images_0] = batch_x_0
                    inp_dict[images_1] = batch_x_1
                    inp_dict[images_2] = batch_x_2
                    inp_dict[labels_one] = batch_y
                    inp_dict[train_phase] = True
                    _, _loss, _acc = sess.run([apply_gradient_op, aver_loss_op, aver_acc_op], inp_dict)

                    avg_loss += _loss
                    avg_acc += _acc
                avg_loss /= total_batch
                avg_acc /= total_batch
                print('Epoch {}:  Train loss {:.4f}, Train acc {:.4f}'.format(epoch, avg_loss, avg_acc))

# 写入tensorboard
                inp_dict_one_epoch = {}
                inp_dict_one_epoch[images_0] = train_imgs
                inp_dict_one_epoch[images_1] = train_imgs[:, start_1:start_1+input_size[1], start_1:start_1+input_size[1], :]
                inp_dict_one_epoch[images_2] = train_imgs[:, start_2:start_2+input_size[2], start_2:start_2+input_size[2], :]
                inp_dict_one_epoch[labels_one] = train_label
                inp_dict_one_epoch[train_phase] = False

                summary_loss, summary_acc = sess.run([loss_visual, acc_visual], inp_dict_one_epoch)
                writer_train.add_summary(summary_loss, epoch)
                writer_train.add_summary(summary_acc, epoch)

#                gpu_info = os.popen('nvidia-smi')
#                print(gpu_info.read())
###########################################
# VALIDATION PART
###########################################
                inp_val_dict = {}
                inp_val_dict[images_0] = imgs_val
                inp_val_dict[images_1] = batch_x_1_val
                inp_val_dict[images_2] = batch_x_2_val
                inp_val_dict[labels_one] = label_val
                inp_val_dict[train_phase] = False
                image_visual = tf.summary.image('x_val', imgs_val, 10)
                _val_loss, _val_acc, summary_val_loss, summary_val_acc, image_show = sess.run([aver_loss_op, aver_acc_op, loss_visual, acc_visual, image_visual], inp_val_dict)
                print('Val loss {:.4f}, Val acc {:.4f}'.format(_val_loss, _val_acc))

# 写入tensorboard
                writer_val.add_summary(image_show, epoch)
                writer_val.add_summary(summary_val_loss, epoch)
                writer_val.add_summary(summary_val_acc, epoch)
                
# =============================================================================
# 调整学习率
# =============================================================================
                if _val_acc_max < _val_acc:
                    _val_acc_max = _val_acc
                    saver.save(sess, os.path.join(path_of_weights, 'model.ckpt'))
                else:
                    _val_patient_pointer += 1

                if _val_patient_pointer >= patient:
                    _val_patient_pointer = 0
                    lr = max(lr*lr_decay_rate, 0.00001)
            print('training DONE.')

single_gpu()











