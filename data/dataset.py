#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:53:35 2019

@author: lierlong
"""

from __future__ import absolute_import
#这句执行之后，/就会变成精确除法，//则代表截断除法
from __future__ import division

import numpy as np
from skimage import transform as sktsf
from data.voc_dataset import vocbboxdataset
from torchvision import transforms as tvtsf
from utils.config import opt
import torch as t
from data.image_transform import resize_bbox,random_flip,flip_bbox


def inverse_normalize(img):
    '''
    逆归一化
    用于将训练的数据可视化
    '''
    if opt.caffe_pretrain:
        img =img+(np.array([122.7717, 115.9465, 102.9801]).reshape(3,1,1))
        return img[::-1, :,:]
    return (img*0.225+0.45).clip(min=0,max=1)*255

def pytorch_normalize(img):
    '''pytorch是在0～1和RGB之间处理图片
    https://github.com/pytorch/vision/issues/223'''
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def caffe_normalize(img):
    '''return -125~125  BGR格式'''
    img = img[[2,1,0], :, :]
    img = img*255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3,1,1)
    img = (img-mean).astype(np.float32, copy = True)
    return img

def preprocess(img, min_size=600, max_size=1000):
    '''
    对图片进行缩放和归一化
    img: CHW和RGB格式的np数组，值的范围0～255
    '''
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H* scale, W* scale),
                       mode='reflect', anti_aliasing = False)
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalize
    return normalize(img), scale
    

class Transform(object):
    '''汇总归一化、图片随机翻转等功能'''
    def __init__(self, min_size = 600, max_size = 1000):
        self.min_size = min_size
        self.max_size = max_size
    
    #输入原始的img、label和bbox，输出变换后的：数据+scale
    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img, scale = preprocess(img, self.min_size, self.max_size)
        _, n_H, n_W = img.shape
        bbox = resize_bbox(bbox,(H,W),(n_H, n_W))
        
        img,params = random_flip(img,x_random=True,return_param=True)
        bbox = flip_bbox(bbox, (n_H,n_W), x_flip=params['x_flip'])
        
        return img, bbox, label, scale

class Dataset:
    def __init__(self,opt):
        self.opt = opt
        self.db = vocbboxdataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)
        
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img,bbox,label))
        return img.copy(),bbox.copy(),label.copy(),scale
    
    def __len__(self):
        return len(self.db)


'''**********返回的ori_img.shape作用是什么****************'''
class TestDataset:
    def __init__(self,opt,split='test',use_difficult=True):
        self.opt = opt
        self.db = vocbboxdataset(opt.voc_data_dir)
        
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img,ori_img.shape[1:],bbox,label,difficult
    
    def __len__(self):
        return len(self.db)
        
        














