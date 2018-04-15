# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:08:11 2018

@author: sy
"""


from glob import glob
import os
from fuzzywuzzy import fuzz
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import sys
#==============================================================================
# 关于label
# 0：表示背景
# 1：脑脊液
# 2：灰质
# 3：白质
#==============================================================================
#==============================================================================
# 首先将原图和label的边缘进行镜像操作，镜像大小为变量，本文为网络最大输入的一半。
# 然后根据这个padding变量来进行数据生成。具体是遍历原图的每一个像素，生成所需训练数据。
#==============================================================================
padding_size = 37
# path_of_goal = '/media/public/0C96AE1147B28ADF/sunyao/MR-train_data'
path_of_goal = '/media/dengy/我的文件/sunyao/out'
num_train = None

def generate_number():
    nowTime=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    randomNum=random.randint(0,1000)
    if int(randomNum) < 10:
        randomNum=str(00)+str(randomNum)
    if 10 <= int(randomNum) < 100:
        randomNum=str(0)+str(randomNum)                
    uniqueNum=str(nowTime)+str(randomNum)
    return uniqueNum


#==============================================================================
# def Mirroring(data):
#     new_data = np.zeros((data.shape[0]+2*padding_size, data.shape[1]+2*padding_size, data.shape[2]))
#     new_data[padding_size:new_data.shape[0]-padding_size, padding_size:new_data.shape[1]-padding_size, :] = data
# #    上边填充        
#     new_data[0:padding_size, padding_size:new_data.shape[1]-padding_size, :] = data[0:padding_size, :,:]
# #    下边填充
#     new_data[new_data.shape[0]-padding_size:new_data.shape[0], padding_size:new_data.shape[1]-padding_size, :] = data[data.shape[0]-padding_size:data.shape[0], :, :]
# #    左边填充
#     new_data[padding_size:new_data.shape[0]-padding_size, 0:padding_size, :] = data[:, 0:padding_size, :]
# #    右边填充
#     new_data[padding_size:new_data.shape[0]-padding_size,new_data.shape[0]-padding_size:new_data.shape[0], ] = data[:, data.shape[1]-padding_size:data.shape[1], :]
# #    左上角填充
#     new_data[0:padding_size, 0:padding_size, :] = data[0:padding_size, 0:padding_size, :]
# #    右上角填充
#     new_data[0:padding_size, new_data.shape[1]-padding_size, :] = data[0:padding_size, data.shape[1]-padding_size, :]
# #    左下角填充
#     new_data[new_data.shape[0]-padding_size:new_data.shape[0], 0:padding_size, :] = data[data.shape[0]-padding_size:data.shape[0], 0:padding_size, :]
# #    右下角填充
#     new_data[new_data.shape[0]-padding_size:new_data.shape[1], new_data.shape[1]-padding_size:new_data.shape[1], :] = data[data.shape[0]-padding_size:data.shape[1], data.shape[1]-padding_size:data.shape[1], :]
#     return new_data
#==============================================================================

def Mirroring(data):
    new_data = np.zeros((data.shape[0]+2*padding_size, data.shape[1]+2*padding_size, data.shape[2]))
    new_data[padding_size:new_data.shape[0]-padding_size, padding_size:new_data.shape[1]-padding_size, :] = data.reshape(data.shape[0], data.shape[1], data.shape[2])
    return new_data
    
# def save_data(x, y, z, image_pad, number):
#     path_of_class_imgs = os.path.join(path_of_goal, num_train, str(number), 'imgs')
#     path_of_class_label = os.path.join(path_of_goal, num_train, str(number), 'label')
#     if not os.path.exists(path_of_class_imgs):
#         os.makedirs(path_of_class_imgs)
#     if not os.path.exists(path_of_class_label):
#         os.makedirs(path_of_class_label)
#     imgs_train_one = image_pad[x-padding_size:x+padding_size+1, y-padding_size:y+padding_size+1, z]
#     data_time = generate_number()
#     label_train_one = np.zeros((1, 4))
#     label_train_one[0, number] = 1
#     imgs_train_one = (imgs_train_one-np.mean(imgs_train_one))/np.std(imgs_train_one)
#     np.save(os.path.join(path_of_class_imgs, data_time+'.npy'), imgs_train_one)
#     np.save(os.path.join(path_of_class_label, data_time+'.npy'), label_train_one)


def save_data(x, y, num_class):
    data_time = generate_number()
    path_of_class_imgs = os.path.join(path_of_goal, num_train, str(num_class))
    path_of_class_label = os.path.join(path_of_goal, num_train, str(num_class))
    if not os.path.exists(path_of_class_imgs):
        os.makedirs(path_of_class_imgs)
    if not os.path.exists(path_of_class_label):
        os.makedirs(path_of_class_label)
    path_of_class_imgs = os.path.join(path_of_goal, num_train, str(num_class), data_time+'_imgs.npy')
    path_of_class_label = os.path.join(path_of_goal, num_train, str(num_class), data_time+'_label.npy')
    np.save(path_of_class_imgs, x)
    np.save(path_of_class_label, y)


def extract_train_data(image_pad, label_pad):
    class_0_img = []
    class_1_img = []
    class_2_img = []
    class_3_img = []
    class_0_label = np.zeros((1, 4))[0, 0] = 1
    class_1_label = np.zeros((1, 4))[0, 1] = 1
    class_2_label = np.zeros((1, 4))[0, 2] = 1
    class_3_label = np.zeros((1, 4))[0, 3] = 1

    pointer_z = True
    for z in range(label.shape[2]):
        if label_pad[:,:,z].max() <= 0:
            continue
        if pointer_z:
            z_old = z
            pointer_z = False

        print(label.shape[2], '横断面层数', z)
#         for x in range(padding_size, label.shape[0]-padding_size):
#             for y in range(padding_size, label.shape[1]-padding_size):
#                 if label[x, y, z] == 0:
#                     save_data(x, y, z, image_pad, 0)
#
#                 if label[x, y, z] == 1:
#                     save_data(x, y, z, image_pad, 1)
# #
#                 if label[x, y, z] == 2:
#                     save_data(x, y, z, image_pad, 2)
#
#                 if label[x, y, z] == 3:
#                     save_data(x, y, z, image_pad, 3)


        for x in range(padding_size, label.shape[0] - padding_size):
            for y in range(padding_size, label.shape[1] - padding_size):
                if label[x, y, z] == 0:
                    imgs_train_one = image_pad[x - padding_size:x + padding_size + 1,
                                     y - padding_size:y + padding_size + 1, z]
                    imgs_train_one = (imgs_train_one - np.mean(imgs_train_one)) / np.std(imgs_train_one)
                    class_0_img.append(imgs_train_one)

                if label[x, y, z] == 1:
                    imgs_train_one = image_pad[x - padding_size:x + padding_size + 1,
                                     y - padding_size:y + padding_size + 1, z]
                    imgs_train_one = (imgs_train_one - np.mean(imgs_train_one)) / np.std(imgs_train_one)
                    class_1_img.append(imgs_train_one)

                if label[x, y, z] == 2:
                    imgs_train_one = image_pad[x - padding_size:x + padding_size + 1,
                                     y - padding_size:y + padding_size + 1, z]
                    imgs_train_one = (imgs_train_one - np.mean(imgs_train_one)) / np.std(imgs_train_one)
                    class_2_img.append(imgs_train_one)

                if label[x, y, z] == 3:
                    imgs_train_one = image_pad[x - padding_size:x + padding_size + 1,
                                     y - padding_size:y + padding_size + 1, z]
                    imgs_train_one = (imgs_train_one - np.mean(imgs_train_one)) / np.std(imgs_train_one)
                    class_3_img.append(imgs_train_one)

        if z - z_old +1 == 10:
            print('save......')
            class_0_img = np.asarray(class_0_img)
            class_1_img = np.asarray(class_1_img)
            class_2_img = np.asarray(class_2_img)
            class_3_img = np.asarray(class_3_img)

            save_data(class_0_img, class_0_label, 0)
            save_data(class_1_img, class_1_label, 1)
            save_data(class_2_img, class_2_label, 2)
            save_data(class_3_img, class_3_label, 3)

            class_0_img = []
            class_1_img = []
            class_2_img = []
            class_3_img = []
            pointer_z = True

        if z == label.shape[2]:
            if len(class_0_img):
                class_0_img = np.asarray(class_0_img)
                save_data(class_0_img, class_0_label, 0)
            if len(class_1_img):
                class_1_img = np.asarray(class_1_img)
                save_data(class_0_img, class_0_label, 1)
            if len(class_2_img):
                class_2_img = np.asarray(class_2_img)
                save_data(class_0_img, class_0_label, 2)
            if len(class_3_img):
                class_3_img = np.asarray(class_3_img)
                save_data(class_0_img, class_0_label, 3)



# path = '/media/public/0C96AE1147B28ADF/dy/data/Cdata/Training_Set'
path = r'/media/dengy/我的文件/sunyao/Training_Set/'
path_every = glob(os.path.join(path, '*'))
for i in path_every:
    if not os.path.isdir(i):
        continue
    for single_data in os.listdir(i):
        # score = fuzz.partial_ratio("fseg", single_data)
        # if score == 100:
        #     label = nb.load(os.path.join(i, single_data)).get_data()
        # else:
        #     image = nb.load(os.path.join(i, single_data)).get_data()
        score_imgs = fuzz.partial_ratio("strip", single_data)
        score_label = fuzz.partial_ratio("segTRI_ana", single_data)
        if score_label == 100:
            label = nb.load(os.path.join(i, single_data)).get_data()
        elif score_imgs == 100:
            image = nb.load(os.path.join(i, single_data)).get_data()
    print(i.split(os.sep)[-1])
    pad_image = Mirroring(image)
    pad_label = Mirroring(label)
    # aaa = pad_label[:,:,100]
    # plt.imshow(pad_label[:,:,100], cmap='gray')
    # plt.show()
#    for zz in range(pad_label.shape[2]):
#        plt.subplot(1,2,1)
#        plt.imshow(pad_image[:,:,zz], cmap='gray')
#        plt.subplot(1,2,2)
#        plt.imshow(pad_label[:,:,zz], cmap='gray')
#        plt.show()
#        print(pad_label[:,:,zz].max())
    num_train = i.split(os.sep)[-1]
    extract_train_data(pad_image, pad_label)

