#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:00:29 2019

@author: lierlong
"""
import numpy as np

def generate_anchor_base(base_size = 16,
                         ratios = [0.5,1,2],
                         anchor_scales = [8,16,32]):
    '''返回一个锚点对应的9个框的坐标，shape：9×4
    base_size用来确定这个锚点的中心，然后用于求x/y_max,x/y_min'''
    py = base_size/2
    px = base_size/2
    anchor_base = np.zeros((len(ratios)*len(anchor_scales),4),dtype=np.float32)
    
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i*len(anchor_scales) +j
            '''从0到3分别代表： y_{min}, x_{min}, y_{max}, x_{max}'''
            anchor_base[index,0] = py - h/2
            anchor_base[index,1] = px - w/2
            anchor_base[index,2] = px + h/2
            anchor_base[index,3] = px + w/2
    return anchor_base
            
def loc2bbox(src_bbox, loc):
    '''{x/y_min,x/y_max}(即src_bbox) —————>
    height,width,xy坐标 +loc里面的变换尺度————>
    真实box里面的height,width,xy坐标
    ————>{x/y_min,x/y_max}(即dst_bbox)'''
    if src_bbox.shape[0] == 0:
        return np.zeros((0,4), dtype=loc.dtype)
    
    src_bbox = src_bbox.astype(src_bbox.dtype,copy=False)
    
    src_height = src_bbox[:,2] - src_bbox[:,0]
    src_width = src_bbox[:,3] - src_bbox[:,1]
    src_ctr_y = src_bbox[:,0] + 0.5*src_height
    src_ctr_x = src_bbox[:,1] + 0.5*src_width
    
    #这种写法最后还是二维的
    dy = loc[:,0::4] 
    dx = loc[:,1::4]
    dh = loc[:,2::4]
    dw = loc[:,3::4]
             
    ctr_y = dy*src_height[:,np.newaxis] + src_ctr_y[:,np.newaxis]
    ctr_x = dx*src_width[:,np.newaxis] +src_ctr_x[:,np.newaxis]
    h = np.exp(dh) * src_height[:,np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    
    dst_bbox = np.zeros(loc.shape, dtype = loc.dtype)
    
    dst_bbox[:,0::4] = ctr_y - 0.5*h
    dst_bbox[:,1::4] = ctr_x - 0.5*w
    dst_bbox[:,2::4] = ctr_y - 0.5*h
    dst_bbox[:,3::4] = ctr_x - 0.5*w
    
    return dst_bbox
    






































