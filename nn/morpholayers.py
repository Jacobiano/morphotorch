import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Union, List

def _se_to_mask(se: torch.Tensor) -> torch.Tensor:
    se_h, se_w = se.size()
    se_flat = se.view(-1)
    num_feats = se_h * se_w
    out = torch.zeros(num_feats, 1, se_h, se_w)
    for i in range(num_feats):
        y = i % se_h
        x = i // se_h
        out[i, 0, x, y] = 1.
    return out

class DepthwiseDilationLayer(nn.Module):
    '''Depthwise Dilation Layer
    '''
    def __init__(self,
                 in_units: int ,
                 kernel_size: Tuple[int, int] = (3, 3)):
           super(DepthwiseDilationLayer, self).__init__()
           self.se_h = kernel_size[0]
           self.se_w = kernel_size[1]
           #Uniform Initialization between -1 to 0.
           self.kernel = nn.Parameter(torch.rand(1,in_units,kernel_size[0]*kernel_size[1],1,1)-1.)
           self.se=_se_to_mask(torch.ones_like(torch.empty(kernel_size)))
           self.register_buffer('se_const', self.se)
           self.pad_d= [self.se_h // 2, self.se_w // 2]

    def forward(self, x):
        #N compute the neighborhood on kernel_size
        N=x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        N=F.conv2d(N, Variable(self.se_const), padding=self.pad_d) 
        N=torch.reshape(N,(x.shape[0],x.shape[1],self.se_h * self.se_w,x.shape[2], x.shape[3]))
        return (N+ self.kernel).max(dim=2)[0]
    

class DepthwiseErosionLayer(nn.Module):
    '''Depthwise Erosion Layer
    '''
    def __init__(self,
                 in_units: int ,
                 kernel_size: Tuple[int, int] = (3, 3)):
           super(DepthwiseErosionLayer, self).__init__()
           self.se_h = kernel_size[0]
           self.se_w = kernel_size[1]
           #Uniform Initialization between -1 to 0.
           self.kernel = nn.Parameter(torch.rand(1,in_units,kernel_size[0]*kernel_size[1],1,1)-1.)
           self.se=_se_to_mask(torch.ones_like(torch.empty(kernel_size)))
           self.register_buffer('se_const', self.se)
           self.pad_d= [self.se_h // 2, self.se_w // 2]
           
    def forward(self, x):
        #N compute the neighborhood on kernel_size
        N=x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        N=F.conv2d(N, Variable(self.se_const), padding=self.pad_d) 
        N=torch.reshape(N,(x.shape[0],x.shape[1],self.se_h * self.se_w,x.shape[2], x.shape[3]))
        return (N+ self.kernel).min(dim=2)[0]