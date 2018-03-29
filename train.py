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



BATCH_SIZE = 8

log_device_placement = False

EPOCH_NUM = 100

DROP_RATE = 0.5
num_output = 4
path_of_data = r'C:\sunyao_document\data\data_out\\'
path_of_weights = r'C:\sunyao_document\data\out\weights'
path_of_log = r'C:\sunyao_document\data\out\log'
every_class_number = 55
#监测一个变量,如果这个变量在patient周期内没有提高,则降低学习率
patient = 2
wd = 0.0001
#下降学习率的比例
lr_decay_rate = 0.7

imgs_all = {}
label_all = {}

#训练集中，验证集比例，将第一个周期的val_rate作为验证集
val_rate = 0.2

#不同分辨率的输入大小
input_size = (25, 51, 75)

#创建一个空类，底下把各种资料都存到类中
class PARAMETERS(object):
    pass


def generate_train_validation_data(train=True):
    one_epochs_imgs = []
    one_epochs_label = []
    for class_i in range(num_output):
        shuffle_index = np.arange(len(imgs_all[class_i]))
        random.shuffle(shuffle_index)
        one_class_imgs = [imgs_all[class_i][i] for i in shuffle_index]
        one_class_label = [label_all[class_i][i] for i in shuffle_index]
        if len(one_class_imgs) <= every_class_number:
            one_epochs_imgs += one_class_imgs
            one_epochs_label += one_class_label
        else:
            one_epochs_imgs += one_class_imgs[:every_class_number]
            one_epochs_label += one_class_label[:every_class_number]

    if not train:
        one_epochs_imgs = one_epochs_imgs[:int(val_rate*len(one_epochs_imgs))]
        one_epochs_label = one_epochs_label[:int(val_rate*len(one_epochs_imgs))]
        for val_data_path in one_epochs_imgs:
            for class_data_path in range(num_output):
                if val_data_path in imgs_all[class_data_path]:
                    imgs_all[class_data_path].remove(val_data_path)
        for val_data_path in one_epochs_label:
            for class_data_path in range(num_output):
                if val_data_path in label_all[class_data_path]:
                    label_all[class_data_path].remove(val_data_path)

    shuffle_index_all = np.arange(len(one_epochs_imgs))
    random.shuffle(shuffle_index_all)
    
    x_train_shuffle = [one_epochs_imgs[i] for i in shuffle_index_all]
    y_train_shuffle = [one_epochs_label[i] for i in shuffle_index_all]
    
    ont_data = np.load(x_train_shuffle[0])
    x_train_data = np.concatenate(tuple([np.load(i).reshape(1, ont_data.shape[0], ont_data.shape[1], 1) for i in x_train_shuffle]), axis=0)
    y_train_data = np.concatenate(tuple([np.load(i) for i in y_train_shuffle]), axis=0)
    return x_train_data, y_train_data


def generate_data_path():
    imgs_path_one = []
    label_path_one = []
    path_all = glob.glob(os.path.join(path_of_data, '*'))
    for singal_class_path in path_all:
        if os.path.isfile(singal_class_path):
            continue
        path_of_imgs = os.path.join(singal_class_path, 'imgs')
        path_of_label = os.path.join(singal_class_path, 'label')
        for every_data_after_clacess in os.listdir(path_of_imgs):
            imgs_path_one.append(os.path.join(path_of_imgs, every_data_after_clacess))
            label_path_one.append(os.path.join(path_of_label, every_data_after_clacess))
        class_number = singal_class_path.split(os.sep)[-1]
        
        for i in range(num_output):
            if class_number == str(i):
                imgs_all[i] = imgs_path_one
                label_all[i] = label_path_one

def single_gpu():
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
        images_0 = tf.placeholder(tf.float32, [None, 25, 25, 1])
        images_1 = tf.placeholder(tf.float32, [None, 51, 51, 1])
        images_2 = tf.placeholder(tf.float32, [None, 75, 75, 1])
        labels_one = tf.placeholder(tf.float32, [None, num_output])
        train_phase = tf.placeholder(tf.bool)
        
        model = segmentation_net.segmentation_model(images_0, images_1, images_2, labels_one, num_output, DROP_RATE)
        
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
# =============================================================================
# 导入数据路径
# =============================================================================     
        generate_data_path()
        
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

# =============================================================================
# 建立可视化
# =============================================================================
            writer_train = tf.summary.FileWriter(os.path.join(path_of_log, 'train'))
            writer_val = tf.summary.FileWriter(os.path.join(path_of_log, 'test'))
            
            writer_train.add_graph(sess.graph)
            merged_all = tf.summary.merge_all()
            # 初始化学习率
            lr = 0.01
            _val_acc_max = 0
            _val_patient_pointer = 0
            for epoch in range(EPOCH_NUM):
# =============================================================================
# 导入数据
# =============================================================================



                if epoch == 0:
                    val_imgs, val_label = generate_train_validation_data(train=False)
                    train_imgs, train_label = generate_train_validation_data(train=True)

                    # val_imgs = images[images.shape[0]-int(images.shape[0]*val_rate):images.shape[0]]
                    # val_label = labels[labels.shape[0]-int(labels.shape[0]*val_rate):labels.shape[0]]
                    # train_imgs = images[:images.shape[0]-int(images.shape[0]*val_rate)]
                    # train_label = labels[:labels.shape[0]-int(labels.shape[0]*val_rate)]
                else:
                    train_imgs, train_label = generate_train_validation_data(train=True)
                    # train_imgs = images
                    # train_label = labels

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
                    assert (batch_x_0.shape[1], batch_x_0.shape[2]) == (input_size[2], input_size[2])
                    
                    start_1 = int((input_size[2]-input_size[1])/2)
                    batch_x_1 = batch_x_0[:, start_1:start_1+input_size[1], start_1:start_1+input_size[1], :]
                    assert (batch_x_1.shape[1], batch_x_1.shape[2]) == (input_size[1], input_size[1])
                    
                    start_2 = int((input_size[2]-input_size[0])/2)
                    batch_x_2 = batch_x_0[:, start_2:start_2+input_size[0], start_2:start_2+input_size[0], :]
                    assert (batch_x_2.shape[1], batch_x_2.shape[2]) == (input_size[0], input_size[0])

                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict[images_0] = batch_x_2
                    inp_dict[images_1] = batch_x_1
                    inp_dict[images_2] = batch_x_0
                    inp_dict[labels_one] = batch_y
                    inp_dict[train_phase] = True
                    _, _loss, _acc, summary_all = sess.run([apply_gradient_op, aver_loss_op, aver_acc_op, merged_all], inp_dict)
# 写入tensorboard
                    writer_train.add_summary(summary_all, epoch*total_batch+batch_idx)
                    
                    avg_loss += _loss
                    avg_acc += _acc
                avg_loss /= total_batch
                avg_acc /= total_batch
                print('Train loss {:.4f}, Train acc {:.4f}'.format(avg_loss, avg_acc))

#                gpu_info = os.popen('nvidia-smi')
#                print(gpu_info.read())
###########################################
# VALIDATION PART
###########################################

                batch_x_0_val = val_imgs[:, start_2:start_2+input_size[0], start_2:start_2+input_size[0], :]
                batch_x_1_val = val_imgs[:, start_1:start_1+input_size[1], start_1:start_1+input_size[1], :]
                batch_x_2_val = val_imgs
                
                inp_val_dict = {}
                inp_val_dict[images_0] = batch_x_0_val
                inp_val_dict[images_1] = batch_x_1_val
                inp_val_dict[images_2] = batch_x_2_val
                inp_val_dict[labels_one] = val_label
                inp_val_dict[train_phase] = False
                image_visual = tf.summary.image('x_val', images, 10)
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











