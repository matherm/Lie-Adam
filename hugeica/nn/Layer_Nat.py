import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter

from .Parameter import *
from .Loss import *

def outer(a, b):
    return torch.bmm(a.unsqueeze(2), b.unsqueeze(1))

def isnan(x):
    return (x != x).sum() > 0

def score_function(p_prime, p):
    return p_prime/p

def relative_gradient(grad_output, fun_output, output):
    psi = score_function(grad_output, fun_output)
    return outer(psi, output)

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
        # G = relative_gradient(grad_output, fun_output, output)
        I = torch.eye(weight.shape[0]).to(weight.device).unsqueeze(0)
        G = I - outer(fun_output , output)
        G = torch.bmm(G, weight.unsqueeze(0).repeat(B, 1, 1))
        G = G.mean(0)
        # K = torch.mean(grad_output, dim=0) - torch.diag(G)
        # G = G * torch.sign(K)
        G = G - G.transpose(2,1)
        return grad_input, G, None

class Relative_SOGradient(nn.Module):

    def __init__(self, n_dims, fun=Loss.Logcosh):
        super().__init__()
        self.weight = SOParameter(torch.eye(n_dims))
        self.fun = fun

    @property
    def components_(self):
        return self.weight

    def forward(self, X):
        return F_SO_Linear_Relative().apply(X, self.weight, self.fun)


class F_Linear_Relative(Function):

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
        I = torch.eye(weight.shape[0]).to(weight.device).unsqueeze(0)
        G = I - outer(fun_output , output)
        G = torch.bmm(G, weight.unsqueeze(0).repeat(B, 1, 1))
        G = G.mean(0)
        #K = torch.mean(grad_output, dim=0) - torch.diag(G)
        #G = G * torch.sign(K)
        return grad_input, G, None


class Relative_Gradient(nn.Module):

    def __init__(self, n_dims, fun=Loss.Logcosh):
        super().__init__()
        self.weight = Parameter(torch.eye(n_dims))
        self.fun = fun

    @property
    def components_(self):
        return self.weight

    def forward(self, X):
        return F_Linear_Relative().apply(X, self.weight, self.fun)