#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 00:20:15 2019

@author: lierlong
"""

import numpy as np
from torch import nn
from torch.nn import functional as F

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator

def normal_init(m,mean,stddev,truncated=False):
    '''
    权重初始化：截断初始化和随机初始化
    '''
    if truncated: #截断
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean,stddev)
        m.bias.data.zero_()
        
def _enumerate_shifted_anchor(anchor_base,feat_stride,height,width):
    '''
    Args:
        anchor:shape=(9,4),特征图里面(1,1)点对应的9个anchor，
               是其他锚点生成框的基础
        feat_stride:原图缩小的倍数
        height/width:特征图的尺寸
    '''
    shift_x = np.arange(0,width*feat_stride,feat_stride)   #shape=(width)
    shift_y = np.arange(0,height*feat_stride,feat_stride)  #shape=(height)
    shift_x,height_y=np.meshgrid(shift_x,shift_y)          #shape=(width,height)
    #生成所有锚点的数量，每个点准备与（9,4)的框相加
    shift = np.stack((shift_y.ravel(),shift_x.ravel(),     #shift.shape=(width*height,4)
                      shift_y.ravel(),shift_x.ravel()),axis=1)
    
    A = anchor_base[0] # A=9
    K = shift.shape[0] #width*height
    #anchor.shape = (k,A,4)
    #TODO:这个数据相加还需要好好理解一下
    anchor = anchor_base.reshape((1,A,4))  +\
             shift.reshape((K,1,4)).transpose((1,0,2))
    anchor = anchor.reshape((K*A,4)).astype(np.float32)
    return anchor
    
class RegionProposalNetwork(nn.Model):
    '''
    本网络用于生成区域提案
    
    Args:
        in/mid_channels:输入和中间层的大小
        ratio:9个框的长宽比
        anchor_scales:9个框的大小
        feat_stride:经过卷积之后特征图缩小的倍数
        proposal_creator_params:用于传入ProposalCreator类的参数
    '''
    def __init__(self,in_channels=512,
                 mid_channels=512,ratios=[0.5,1,2],
                 anchor_scales=[8,16,32],feat_stride=16,
                 proposal_creator_params=dict()):
        super(RegionProposalNetwork,self).__init__()
        self.anchor_base=generate_anchor_base(anchor_scales=anchor_scales,
                                              ratios=ratios)
        self.feat_stride=feat_stride
        #TODO:为什么要传入这个self变量
        self.proposal_layer=ProposalCreator(self,**proposal_creator_params)
        n_anchor=self.anchor_base.shape[0] #9
        self.conv1=nn.Conv2d(in_channels,mid_channels,3,1,1) #卷积核3*3
        self.score=nn.Conv2d(mid_channels,n_anchor*2,1,1,0)  #卷积核 1*1,是不是物体二分类
        self.loc=nn.Conv2d(mid_channels,n_anchor*4,1,1,0)    #卷积核1*1,回归框的4个坐标
        #TODO:为什么要这样初始化,以及bais要置零
        normal_init(self.conv1,0,0.01)
        normal_init(self.score,0,0.01)
        normal_init(self.loc,0,0.01)
    
    def forward(self,x,img_size,scale=1.):
        '''
        Args:
            x:经过卷积之后的特征图
            img_size:图片原始长宽
        return:
            返回5个量，数据类型依次是：
            (t.autograd.Variable,t.autograd.Variable,array,array,array)
            rpn_locs:用于预测bbox的偏移和缩放，shape=(N,H W A,4)
            rpn_scores:bbox是前景的概率，shape=(N,H W A,2)
            rois:bbox的坐标，shape=(R,4)
            
        '''
        n,_, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base),
                                           self.feat_stride,hh,ww)
        n_anchor = anchor.shape[0] // (hh*ww)
        
        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)      #生成坐标
        rpn_scores = self.score(h)  #生成是否是前景的概率
        
        #permute是对不同维度的数据调换位置，view作用类似reshape，要求内存必须整块
        #所以一般用contiguous把内存整合成一块
        rpn_locs = rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)
        rpn_scores = rpn_scores.permute(0,2,3,1).contiguous()
        #TODO:hh和ww不会导致数据不对劲吗
        rpn_softmax_scores=F.softmax(rpn_scores.view(n,hh,ww,n_anchor,2),dim=4)
        rpn_fg_scores = rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n,-1)
        rpn_scores = rpn_scores.view(n,-1,2)
        
        rois = list()
        #TODO:这是个什么变量
        roi_indices = list()
        #经过网络每一个点都会产生一个概率和坐标变换，将网络输出通过ProposalCreator函数用
        #nms进行筛选，得到rois
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(),
                                      rpn_fg_scores[i].cpu().data.numpy(),
                                      anchor,img_size,scale = scale)
            batch_index=i*np.ones((len(roi),),dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        
        rois = np.concatenate(rois,axis=0)
        roi_indices = np.concatenate(roi_indices,axis=0)
        return rpn_locs,rpn_scores,rois,roi_indices,anchor
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    