import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter

from .Parameter import *
from .Loss import *

def isnan(x):
    return (x != x).sum() > 0

def score_function(p_prime, p):
    return p_prime/p

def relative_gradient(grad_output, fun_output, output):
    psi = score_function(grad_output, fun_output)
    return psi.T @ output 

class F_SO_Linear_Relative(Function):

    @staticmethod
    def forward(ctx, inpt, weight, fun):
        outpt = inpt.mm(weight.T)
        ctx.save_for_backward(inpt, weight, outpt, fun(outpt))
        return outpt

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, output, fun_output = ctx.saved_tensors
        B = inpt.shape[0]
        grad_input = grad_weight = None
        # Gradient w.r.t. input
        # grad_input = grad_output.mm(weight) # gradient is not needed in previous layers
        # Gradient w.r.t. weights
        G = relative_gradient(grad_output, fun_output, output)
        # K = torch.mean(grad_output, dim=0) - torch.diag(G)
        # G = G * torch.sign(K)
        grad_rel = G - G.T
        return grad_input, grad_rel, None

class Relative_Gradient(nn.Module):

    def __init__(self, n_dims, fun=Loss.Logcosh):
        super().__init__()
        self.weight = SOParameter(torch.eye(n_dims))
        self.fun = fun

    @property
    def components_(self):
        return self.weight

    def forward(self, X):
        return F_SO_Linear_Relative().apply(X, self.weight, self.fun)