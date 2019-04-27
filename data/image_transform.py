#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:19:14 2019

@author: lierlong
"""
from PIL import Image
import numpy as np

def read_image(path , dtype = np.float32):
    '''
    读取图片，返回图片的np.ndarray
    np读取的图片格式是height * width * channel
    pytorch需要转换成chw , tensorflow保持不变
    '''
    with Image.open(path) as file:
        img = np.asarray(file , dtype = dtype)
    
    #reshape (h,w) -->  (1,h,w)
    if img.ndim == 2:
        return img[np.newaxis,:] #np.newaxis给所在的位置增加1维度
    else:
        #reshape (h,w,c) --> (c,h,w)
        return img.transpose((2,0,1))
    
    
'''*******************************
********变换的不变点怎么定的******
**********************************
'''
def resize_bbox(bbox , in_size , out_size):
    '''
    返回重新变换bbox框大小的宽高
    bbox : 输入格式是二维数组，0维是图片编号，1维是`(y_{min}, x_{min}, y_{max}, x_{max})`
    in/out_size : 是(高，宽)
    '''
    new_bbox = bbox
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    new_bbox[:,0] = bbox[:,0] * y_scale
    new_bbox[:,1] = bbox[:,1] * y_scale
    new_bbox[:,2] = bbox[:,2] * x_scale
    new_bbox[:,3] = bbox[:,3] * x_scale
    return new_bbox

    
'''***************************************
********怎么翻转的，编号是不是搞错了******
******************************************'''    
def flip_bbox(bbox, size, y_flip = False, x_flip= False):
    '''翻转bbox
    args:
        bbox: 同resize_bbox
        size: (H,W)
        y_flip： 垂直翻转
        x_flip: 水平翻转
    '''
    H, W = size
    flip_bbox = bbox
    if y_flip:
        y_min = H - bbox[:,0]
        y_max = H - bbox[:,2]
        flip_bbox[:, 0] = y_min
        flip_bbox[:, 2] = y_max
    if x_flip:
        x_min = H - bbox[:,1]
        x_max = H - bbox[:,3]
        flip_bbox[:, 1] = x_min
        flip_bbox[:, 3] = x_max
    return flip_bbox





























