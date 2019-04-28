#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:40:18 2019

@author: lierlong

实现对原始信息的提取，并未作处理
image_transform: 数据变换(翻转等)
dataset: 实现面向网络的变换
"""
import os
import numpy as np
import xml.etree.ElementTree as ET
from .image_transform import read_image


voc_bbox_label_names = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

class vocbboxdataset:
    '''
    读取xml文件里面的bbox、图片、label等信息
    数据下载:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    
    '''
    def __init__(self, data_dir, split='trainval', use_difficult=False,
                 return_difficult = False):
        id_list_file = os.path.join(
                        data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = voc_bbox_label_names
        self.use_difficult = use_difficult
    
    def __len__(self):
        return len(self.ids)
    
    def get_example(self, i):
        '''args:
            i:图片的序号'''
        id_ = self.ids[i]
        #用ET这个库读取xml信息
        anno = ET.parse(
                os.path.join(self.data_dir,'Annotations', id_+'.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text)-1 
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.float32)
        #different是0和1的数组，转换为布尔值，true或者false
        difficult = np.array(difficult, dtype = np.bool)
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_+'.jpg')
        img = read_image(img_path)
        
        return img, bbox, label, difficult
    
    __getitem__ = get_example
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
