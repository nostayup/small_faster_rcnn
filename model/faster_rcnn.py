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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    