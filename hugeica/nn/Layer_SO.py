import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter

from .Parameter import *
from .Loss import Loss

def isnan(x):
    return (x != x).sum() > 0

def lie_bracket(weight, grad_weight):
    lie_bracket = grad_weight.mm(weight.T) - weight.mm(grad_weight.T)
    return lie_bracket

class F_SO_Linear(Function):

    @staticmethod
    def forward(ctx, inpt, weight):
        outpt = inpt.mm(weight.T)
        ctx.save_for_backward(inpt, weight, outpt)
        return outpt

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, outpt = ctx.saved_tensors
        B = inpt.shape[0]
        grad_input = grad_weight = None
        # Gradient w.r.t. input
        # grad_input = grad_output.mm(weight) # gradient is not needed in previous layers
        # Gradient w.r.t. weights
        grad_weight = grad_output.T.mm(inpt)
        grad_lie = lie_bracket(weight, grad_weight)
        return grad_input, grad_lie

class SO_Layer(nn.Module):

    def __init__(self, n_dims):
        super().__init__()
        self.weight = SOParameter(torch.eye(n_dims))
        self.reset_()

    def reset_(self):
        self.weight.data = torch.qr(torch.distributions.Normal(0,0.01).sample(self.weight.data.shape))[0]

    @property
    def components_(self):
        return self.weight

    def forward(self, X):
        return F_SO_Linear().apply(X, self.weight)

class F_SO_2d(Function):

    @staticmethod
    def forward(ctx, inpt, weight, n_components, filter_size, stride):
        B, C, H, W = inpt.shape
        # rollout the weights
        kernel = weight.view(n_components, C, filter_size, filter_size).clone()
        ctx.save_for_backward(inpt, weight, kernel, stride)
        outpt = F.conv2d(inpt, kernel, stride=stride.item()) # (B, n_components, H, W)
        return outpt

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, kernel, stride = ctx.saved_tensors
        grad_input = grad_weight = grad_lr = grad_n_components = grad_filter_size = None
        # Gradient w.r.t. input
        grad_input = F.grad.conv2d_input(grad_output=grad_output, weight=kernel, stride=stride.item(), input_size=inpt.shape)
        # Gradient w.r.t. weights
        grad_weight = F.grad.conv2d_weight(grad_output=grad_output, input=inpt, stride=stride.item(), weight_size=kernel.shape)
        # unroll the weights
        grad_weight = grad_weight.view(weight.shape).contiguous()
        grad_lie = lie_bracket(weight, grad_weight)
        return grad_input, grad_lie, None, None, None

class SO_Layer2d(nn.Module):

    def __init__(self, in_channels, filter_size, stride=1):
        super().__init__()
        self.n_components = in_channels*filter_size*filter_size
        self.weight = SOParameter(torch.eye(self.n_components))
        self.filter_size = filter_size
        self.stride = torch.ones(1).int() * stride
        self.reset_()

    def reset_(self):
        self.weight.data = torch.qr(torch.distributions.Normal(0,0.01).sample(self.weight.data.shape))[0]

    @property
    def components_(self):
        return self.weight

    def forward(self, X):
        """
        Args:
            X : input tensor with shape (B, C, H, W)
        """
        assert X.ndim == 4
        return F_SO_2d().apply(X, self.weight, self.n_components, self.filter_size, self.stride)