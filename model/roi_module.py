#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:13:19 2019

@author: lierlong
"""

from collections import namedtuple  #用于生成一个特定的字典，namedtuple(dict_name,[属性1,属性2,……])
from string import Template  #将字符串中的某些量用变量替换掉

import cupy as cp
import torch as t
from torch.autograd import Function

#TODO:roi_cupy需要注释一下
from model.utils.roi_cupy import kernel_forward,kernel_backward

Stream = namedtuple('Stream',['ptr'])

#TODO:这个装饰器的作用
@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

cuda_num_threads = 1024

def get_blocks(n,k=cuda_num_threads):
    return (n+k-1)//k


#TODO:主要部分都是在roi_cupy中完成，需要完善。
class RoI(Function):
    def __init__(self,outh,outw,spatial_scale):
        self.forward_fn = load_kernel('roi_forward',kernel_forward)
        self.backward_fn = load_kernel('roi_backward',kernel_backward)
        self.outh = outh
        self.outw = outw
        self.spatial_scale = spatial_scale
        
    def forward(self, x, rois):
        #x是经过卷积的特征图
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = B,C,H,W = x.size()
        self.N = N = rois.size(0)
        self.argmax_data = t.zeros(N,C,self.outh,self.outw).int().cuda()
        self.rois = rois
        
        output = t.zeros(N,C,self.outh,self.outw).cuda()
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale, C,H,W,
                self.outh,self.outw,
                output.numel()]
        stream = Stream(ptr=t.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                        block=(cuda_num_threads,1,1),
                        grid=(get_blocks(output.numel()),1,1),
                        stream = stream)
        return output
    
    def backward(self,grad_output):
        
        grad_output=grad_output.contiguous()
        B,C,H,W = self.in_size
        grad_input = t.zeros(self.in_size).cuda()
        stream = Stream(ptr=t.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                self.N, self.spatial_scale, C,H,W,
                self.outh,self.outw,
                grad_input.numel()]
        self.backward_fn(args=args,
                        block=(cuda_num_threads,1,1),
                        grid=(get_blocks(grad_input.numel()),1,1),
                        stream = stream)
        return grad_input,None


class RoIPooling2D(t.nn.Module):
    def __init__(self,outh,outw,spatial_scale):
        super(RoIPooling2D,self).__init__()
        self.RoI = RoI(outh,outw,spatial_scale)
    def forward(self,x,rois):
        return self.RoI(x,rois)
    
    


































