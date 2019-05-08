#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:20:41 2019

@author: lierlong
"""
'''
本脚本实现数据类型转换:
    cpu转gpu使用t.cuda() 
    gpu转cpu使用t.cpu() 
    cpu,gpu转variable使用Variable(t) 
    Variable转cpu，gpu使用v.data 
    tensor转numpy使用t.numpy() 
    numpy转tensor使用torch.from_numpy()
'''


import numpy as np
import torch as t
#detach():作用是对该变量不进行求导操作

def tonumpy(data):
    if isinstance(data,np.ndarray):
        return data
    if isinstance(data,t.Tensor):
        return data.detach().cpu().numpy()
    
def totensor(data, cuda=True):
    if isinstance(data,np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data,t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

def scalar(data):
    if isinstance(data,np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()

