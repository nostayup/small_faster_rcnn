#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:28:21 2019

@author: lierlong
"""

class config:
    '''data'''
    voc_data_dir ='/home/lierlong/code/simple_faster_rcnn_bylerl/VOCdevkit/VOC2007/'
    min_size = 600
    max_size = 1000  #image resize 
    num_works = 8
    test_num_workers = 8
    
    caffe_pretrain = False
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
    
    lr = 1e-3
    lr_decay = 0.1
    weight_decay = 0.0005
    
    
opt = config
    