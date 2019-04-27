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
from .voc_dataset import vocbboxdataset
from torchvision import transforms as tvtsf
from utils.config import opt
import torch as t

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
    return normalize(img)
    