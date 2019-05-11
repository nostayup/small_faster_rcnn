#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:08:18 2019

@author: lierlong
"""
from __future__ import absolute_import
from __future__ import division

import torch as t
import numpy as np
import cupy as cp
from torch import nn
from torch.nn import functional as F

from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression
from data.dataset import preprocess
from utils.config import opt


def nograd(f):
    '''在正向传播时，把求导关掉'''
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)
    return new_f

        


class FasterRCNN(nn.Module):
    '''
    Faster_RCNN主要由以下三个阶段构成：
        1.feature:特征图计算
        2.RPN：通过特征图产生roi
        3.head:头部网络：通过计算ROI的特征图进行分类和框回归
            
    Args:
        extractor(nn.Module):读取图片，进行卷积
        rpn(nn.Module):生成roi
        head(nn.Module):读取shape=BCHW的Variable，RoIs及其编号。返回位置参数和分类得分
        loc_normalize_mean(type=tuple):位置估计平均值
        loc_normalize_std(type=tuple):位置估计偏差
    '''
    def __init__(self,extractor, rpn, head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std=(0.1,0.1,0.2,0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')
        
    @property
    def n_class(self):
        #Total numbel of class including the backgrand
        return self.head.n_class
    
    def forward(self, x, scale=1.):
        '''
        Args:
            x:原始图像
            scale:图像放缩倍数
        Return:
            返回4个值，类型分别是：Variable,Variable,array,array
            roi_cls_locs:RoI框的平移和缩放
            roi_scores:每个ROI的物体类别得分
            rois:RPN网络生成的roi
            roi_indices:ROI的batch编号
        '''
        img_size = x.shape[2:] #height*width
        
        h = self.extractor(x) #特征图计算网络
        #TODO:rpn_locs的作用
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h,img_size,scale)
        roi_cls_locs,roi_scores = self.head(h,rois,roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices
    
    def use_preset(self, preset):
        '''
        设置用于确定nms和舍弃一些框的阀值
        '''
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')
    
    @nograd
    def predict(self, imgs, sizes = None, visualize = False):
        '''
        对每张图片进行预测，
        Args:
            输入图片必须是CHW格式的RGB，是np.ndarry
        Return:
            返回的是一个tuple，包含：框的坐标，标签，得分
            (bboxes,labels,scores)
        '''
        self.eval()
        if visualize:  #可视化
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]  #get width&height
                #TODO:为什么可视化需要随机处理
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = []
        scores = []
        for img,size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            #TODO:调用forward函数，为什么可以这么调用
            roi_cls_loc,roi_scores,rois,_ =self(img, scale=scale) 
            #TODO:.data是什么作用
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale
            
            mean = t.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]
            
            roi_cls_loc = (roi_cls_loc*std + mean)
            #TODO: 这个会有变形的作用吗
            roi_cls_loc = roi_cls_loc.view(-1,self.n_class,4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1,4)),
                                at.tonumpy(roi_cls_loc).reshape((-1,4)))
            cls_bbox = at.totensor((cls_bbox))
            cls_bbox = cls_bbox.view(-1,self.n_class*4)
            
            '''clamp表示将tensor限制在其范围，让框不超过图片'''
            cls_bbox[:,0::2] = (cls_bbox[:,0::2]).clamp(min=0, max=size[0])
            cls_bbox[:,1::2] = (cls_bbox[:,1::2]).clamp(min=0, max=size[1])
            
            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))
            
            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            
            bbox,label,score = self._suppress(raw_cls_bbox,raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        self.use_preset('evaluate')
        self.train()
        return bboxes,labels,scores

    
    def _suppress(self, raw_cls_bbox,raw_prob):
        '''之前的数据是包含了所有21个类别的得分，
        把无效数据去除'''
        bbox = []
        label = []
        score = []
        for i in range(1,self.n_class):    #0是背景，去除掉
            cls_bbox_i = raw_cls_bbox.reshape((-1,self.n_class,4))[:,i,:] #去除第i个类别对应的框
            prob_i = raw_prob[:,i]    #第i个类别的得分
            
            mask = prob_i > self.score_thresh #mask是Ture/False
            cls_bbox_i = cls_bbox_i[mask]  #如果是False，就返回空数组,true就返回原数组
            prob_i = prob_i[mask]
            keep = non_maximum_suppression(cp.array(cls_bbox_i), self.nms_thresh,prob_i)
            keep = cp.asnumpy(keep)
            
            bbox.append(cls_bbox_i[keep])
            label.append((i-1)*np.ones((len(keep),)))
            
            score.append(prob_i[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.float32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
    
    def get_optimizer(self):
        
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params +=[{'params':[value], 'lr':lr*2, 'weight_decay':0}]
                else:
                    params += [{'params':[value],'lr':lr,'weight_decay':opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer
    
    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *=decay
        return self.optimizer
    
        
        

        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    