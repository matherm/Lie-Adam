import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from .Variance_Layer import OnlineVariance

EPS = 1e-7

class DeltaRule(Function):

    @staticmethod
    def forward(ctx, inpt, weight):
        norm = torch.norm(weight, dim=1).reshape(-1, 1)
        # needs to be done in-place, otherwise pytorch caches the wrong weights.
        weight /= norm 
        ctx.save_for_backward(inpt, weight)
        return inpt.mm(weight.t())

    @staticmethod
    def naive(weight, X):
        grad_weight = torch.zeros_like(weight, dtype=weight.dtype, device=weight.device)
        for _x in X:
            x = torch.unsqueeze(_x, dim=1)
            y = torch.matmul(weight, x)
            yyT = torch.matmul(y, torch.t(y))
            yxT = torch.matmul(y, torch.t(x))
            grad_weight += yxT - torch.matmul(torch.tril(yyT), weight)
        return grad_weight
        
    @staticmethod
    def improved(weight, X):
        y = torch.einsum('of,bf->bo', weight, X)
        X = X.unsqueeze(2) # (b, f, 1)
        y = y.unsqueeze(2) # (b, o, 1)
        yyT = torch.bmm(y, y.transpose(1,2)) # ( b , o, o)
        yxT = torch.bmm(y, X.transpose(1,2)) # ( b , o, f) 
       
        tril_mask = torch.tril(torch.ones((weight.size(0), weight.size(0)), dtype=weight.dtype, device=weight.device))  # (o, o)  
        yyT.mul_(tril_mask.unsqueeze(0)) # (b, o, o)  
        q = torch.bmm(yyT, weight.expand(yxT.size()))  # (b, o, o), (1, o, f) > (b, o, f)

        grad_weight = torch.sum(yxT.sub_(q), dim=0) # (o, f)
        return grad_weight

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight= ctx.saved_tensors
        
        grad_input = grad_weight = None
        
        # Gradient w.r.t. input
        grad_input = grad_output.mm(weight)
        
        # Gradient w.r.t. weights
        grad_weight = DeltaRule.improved(weight, inpt)
        
        # Testing
        # grad_weight = DeltaRule.naive(weight, inpt)
        # assert torch.norm(grad_weight - DeltaRule.naive(weight, inpt)) < 1e-3
        return grad_input, -grad_weight

class HebbianLayer(nn.Module):

    def __init__(self, in_features, out_features, ds_size=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.online_var = OnlineVariance(out_features, ds_size)

        self.is_converged = False
        self.reset_parameters()

    def reset_parameters(self, l=-0.1, u=0.1):
        _W = torch.randn(self.out_features, self.in_features)
        _W = (_W - torch.min(_W)) / (torch.max(_W) - torch.min(_W)) * (u - l) + l
        norm = torch.norm(_W, dim=1).reshape(-1, 1)
        self.weight.data = _W / norm

    def has_converged(self):
        if hasattr(self, "old_weights"):
            eps = torch.pow(self.weight.detach() - self.old_weights, 2).sum()
            if eps < EPS: 
                self.is_converged = True
        self.old_weights = self.weight.clone().detach()
        return self.is_converged
    
    @property    
    def sphering_matrix(self):
        w = self.weight.detach()
        w = w / torch.norm(w, dim=1, keepdim=True).reshape(-1, 1)
        w[torch.isnan(w)] = 0.
        W = w.T  
        return W / self.online_var.std.detach()
    
    def step(self, ep, lr, param_group):
        self.has_converged()
        param_group['lr'] = lr / np.sqrt(1 + 1 + ep) 

    def forward(self, X):
        X = DeltaRule.apply(X, self.weight)
        X = self.online_var(X)
        if self.is_converged:
            return X.detach()
        else:
            return X