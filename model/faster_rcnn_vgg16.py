#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:44:46 2019

@author: lierlong
"""
from torch import nn
import torch
from torchvision.models import vgg16
from model.roi_module import RoIPooling2D
from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork
from utils import array_tool as at
from utils.config import opt


def normal_init(m, mean, stddev, truncated = False):
    '''截断初始化和随机初始化'''
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean,stddev)
        m.bias.data.zero_()

class VGG16RoIHead(nn.Module):
    '''
    通过特征图和和rpn给出的ROI，输出框的位置和类别分类
    Args:
        n_class(int):包含背景在内的分类类别数量
        roi_size(int):ROI_pooling之后的特征图的height和width
        spatial_scale(float):roi进行resize的比例
        classifier(nn.Module):vgg16里面的两层网络的分类器
    '''
    def __init__(self,n_class,roi_size,spatial_scale,classifier):
        super(VGG16RoIHead,self).__init__()
        
        self.classifier = classifier
        #TODO:为什么没有把背景的类别去掉
        self.cls_loc = nn.Linear(4096, n_class*4)
        self.score = nn.Linear(4096,n_class)
        normal_init(self.cls_loc,0,0.001)
        normal_init(self.score,0, 0.01)
        
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        #构建了一个roi生成网络
        self.roi = RoIPooling2D(self.n_class, self.roi_size, self.spatial_scale)
        
    def forward(self, x, rois, roi_indices):
        '''
        Args:
            x(Variable): batch*channel*h*w 的图像数据
            rois(Tensor):一组图像roi的边框坐标
            roi_indices(Tensor):一组图片里面的bbox的序号
        '''
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:,None],rois],dim=1)
        #TODO:数据结构需要搞清楚
        xy_indices_and_rois = indices_and_rois[:,[0,2,1,4,3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
    
class FasterRCNNVGG16(FasterRCNN):
    
        
        
        
        
        
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        