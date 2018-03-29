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
path_of_goal = '/media/public/0C96AE1147B28ADF/sunyao/MR-train_data'


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
    new_data[padding_size:new_data.shape[0]-padding_size, padding_size:new_data.shape[1]-padding_size, :] = data
    return new_data
    
def save_data(x, y, z, image_pad, number):
    path_of_class_imgs = os.path.join(path_of_goal, str(number), 'imgs')
    path_of_class_label = os.path.join(path_of_goal, str(number), 'label')
    if not os.path.exists(path_of_class_imgs):
        os.makedirs(path_of_class_imgs)
    if not os.path.exists(path_of_class_label):
        os.makedirs(path_of_class_label)
    imgs_train_one = image_pad[x-padding_size:x+padding_size+1, y-padding_size:y+padding_size+1, z]
    data_time = generate_number()
    label_train_one = np.zeros((1, 4))
    label_train_one[0, number] = 1
    imgs_train_one = (imgs_train_one-np.mean(imgs_train_one))/np.std(imgs_train_one)
    np.save(os.path.join(path_of_class_imgs, data_time+'.npy'), imgs_train_one)
    np.save(os.path.join(path_of_class_label, data_time+'.npy'), label_train_one)

    
    
def extract_train_data(image_pad, label_pad):
    for z in range(label.shape[2]):
        if label_pad[:,:,z].max() <= 0:
            continue
        print('$$$$$$$$$$$$$')
        print(z)
        for x in range(padding_size, label.shape[0]-padding_size):
            for y in range(padding_size, label.shape[1]-padding_size):
                if label[x, y, z] == 0:
                    save_data(x, y, z, image_pad, 0)
    
                if label[x, y, z] == 1:
                    save_data(x, y, z, image_pad, 1)
#                    
                if label[x, y, z] == 2:
                    save_data(x, y, z, image_pad, 2)    

                if label[x, y, z] == 3:
                    save_data(x, y, z, image_pad, 3)  



path = '/media/public/0C96AE1147B28ADF/dy/data/Cdata/Training_Set'
path_every = glob(os.path.join(path, '*'))
for i in path_every:
    if not os.path.isdir(i):
        continue
    for single_data in os.listdir(i):
        score = fuzz.partial_ratio("fseg", single_data)
        if score == 100:
            label = nb.load(os.path.join(i, single_data)).get_data()
        else:
            image = nb.load(os.path.join(i, single_data)).get_data()
    print(i, '第几个文件夹')
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
    extract_train_data(pad_image, pad_label)

