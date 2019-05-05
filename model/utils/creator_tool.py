#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:17:48 2019

@author: lierlong
"""

import numpy as np
import cupy as cp

from bbox_tools import loc2bbox
from nms import non_maximum_suppression


class ProposalCreator:
    '''通过该class生成提议框
    
    Args:
        nms_thresh:nms的阀值
        n_train_pre_nms:排名前多少的框传递给nms用于训练
        n_train_post_nms:排名后多少的框传递给nms用于训练
        n_test_pre_nms:排名前多少的框传递给nms用于测试
        n_test_post_nms:排名后多少的框传递给nms用于测试
        '''
    def __init__(self,
                 parent_model,
                 nms_thresh = 0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.parent_model=parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    def __call__(self,loc,score,anchor,img_size,scale=1.):
        '''
        
        Args:
            loc:锚点产生的bbox对应到真bbox的偏移和缩放，shape=(R,4)
            score:锚点对应的是前景的可能性,shape=(R,)
            anchor:锚点产生的bbox的x/y_min/max四个坐标，shape=(R,4)        
            img_size:图片的原始尺寸，（height，width）
            scale:对图片的缩放
            
        return:
            一组生成提议框的坐标，（s,4)，数量不超过n_test/train_post_nms
        '''
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        #将生成的锚点框平移缩放成roi
        roi = loc2bbox(anchor,loc)
        #将超出图片y坐标的上下限重置为0和img_size[0]
        roi[:,slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)],0,img_size[0])
        #将超出图片x坐标的上下限重置为0和img_size[1]
        roi[:,slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)],0,img_size[0])
        
        min_size = self.min_size*scale
        hs = roi[:,2] - roi[:,0]
        ws = roi[:,3] - roi[:,1]
        #shape=(np.array[],),所以通过[0]把数组提取出来，这里是要保留的数据
        keep = np.where((hs>=min_size)&(ws>=min_size))[0] 
        roi = roi[keep,:]
        score = score[keep]
        
        #ravel()将函数铺平成一维，argsort()将函数从小到大排序后，返回排序后
        #值在原数组里面的序号
        order = score.ravel().argsort()[::-1]
        if n_pre_nms>0:
            order = order[:n_pre_nms]
        roi = roi[order,:]
        
        # TODO: apply nms here(eg:threshoid=0.7)
        keep = non_maximum_suppression(
                cp.ascontiguousarray(cp.asarray(roi)),
                thresh=self.nms_thresh)
        if n_post_nms>0:
            keep=keep[:,n_post_nms]
        roi = roi[keep]
        return roi
            


























