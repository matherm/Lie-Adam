import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter

from .SO_Parameter import *

def isnan(x):
    return (x != x).sum() > 0

def lie_bracket(weight, grad_weight):
    lie_bracket = grad_weight.mm(weight.T) - weight.mm(grad_weight.T)
    return lie_bracket

class F_SO_Linear(Function):

    @staticmethod
    def forward(ctx, inpt, weight):
        ctx.save_for_backward(inpt, weight)
        outpt = inpt.mm(weight.T)
        return outpt

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight = ctx.saved_tensors
        B = inpt.shape[0]
        grad_input = grad_weight = None
        # Gradient w.r.t. input
        grad_input = grad_output.mm(weight)
        # Gradient w.r.t. weights
        grad_weight = grad_output.T.mm(inpt)
        grad_lie = lie_bracket(weight, grad_weight)
        return grad_input, grad_lie

class SO_Layer(nn.Module):

    def __init__(self, n_dims):
        super().__init__()
        self.weight = SOParameter(torch.eye(n_dims))

    def forward(self, X):
        return F_SO_Linear().apply(X, self.weight)