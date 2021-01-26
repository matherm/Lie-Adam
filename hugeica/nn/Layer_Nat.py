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

def relative_gradient(grad_output, y, u):
    psi = score_function(grad_output, y)
    return outer(psi, u)

class F_Linear_Relative(Function):

    @staticmethod
    def forward(ctx, inpt, weight, fun):
        outpt = inpt.mm(weight.T)
        ctx.save_for_backward(inpt, weight, outpt, fun(outpt))
        return outpt

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, output, y = ctx.saved_tensors
        B = inpt.shape[0]
        grad_input = grad_weight = None
        # Gradient w.r.t. input
        # grad_input = grad_output.mm(weight) # gradient is not needed in previous layers
        # Gradient w.r.t. weights
        I = torch.eye(weight.shape[0]).to(weight.device)
        G = I - (y.T @ output)
        G = G @ weight
        K = torch.mean(grad_output, dim=0, keepdims=True).detach() - torch.diag(G).detach()
        G = G * torch.sign(K).detach()
        return grad_input, -G/B, None


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
