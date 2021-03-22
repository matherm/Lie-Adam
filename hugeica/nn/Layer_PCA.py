import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from ..nn import NoGDParameter
from ..helpers import *


def batch_pca_update(inpt, weight, S, mean_, var_, ups_ds_size, updating, n_components_, n_samples, n_samples_seen_):
    if n_components_ > len(inpt):
        warnings.warn(f"Skipping backward(). n_components={n_components_} must be less or equal to the batch number of samples {len(inpt)}")

    col_mean, col_var, n_total_samples =  incremental_mean_and_var(
                                                        inpt, last_mean=mean_, last_variance=var_,
                                                        last_sample_count=n_samples_seen_)
    # Build matrix of combined previous basis and new data
    if n_samples_seen_ == 0:
        inpt = inpt - col_mean 
    else:
        col_batch_mean = torch.mean(inpt, axis=0, keepdim=True)
        inpt = inpt - col_batch_mean 
        mean_correction =  torch.sqrt((n_samples_seen_ * n_samples) / n_total_samples) * (mean_ - col_batch_mean)
        inpt = torch.cat([S.view((-1, 1)) * weight, inpt, mean_correction], dim=0)

    # Update
    device = inpt.device
    # U, S, V = torch.svd(inpt, some=True) # svd returns V instead of numpy-ish V.T
    U, S, V = torch.svd_lowrank(inpt, q=min(inpt.shape))
    U, S, V = U.to(device), S.to(device), V.to(device) 
    U, V =  svd_flip(U, V.T)
    explained_variance = S ** 2 / (n_total_samples - 1) 

    ups_ds_size.data[0] = n_total_samples
    new_weight = V[:n_components_]
    new_S = S[:n_components_]
    new_explained_var = explained_variance[:n_components_]
    new_mean_ = col_mean
    new_var_ = col_var
    return new_weight, new_S, new_explained_var, new_mean_, new_var_

class F_Batch_PCA(Function):
    """
    Adopted from https://github.com/scikit-learn/scikit-learn/blob/483cd3eaa/sklearn/decomposition/_incremental_pca.py
    """

    @staticmethod
    def forward(ctx, inpt, weight, S, explained_var, mean_, var_, ups_ds_size, updating):
        ctx.save_for_backward(inpt, weight, S, explained_var, mean_, var_, ups_ds_size, updating)
        return  (inpt - mean_).mm(weight.T)  / torch.sqrt(explained_var)
        
    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, S, explained_var, mean_, var_, ups_ds_size, updating = ctx.saved_tensors
        # Gradient w.r.t. input
        grad_input = grad_output.mm(weight)
        
        # Gradient w.r.t. weights # Gradient w.r.t. bias
        n_components_ = weight.shape[0]
        n_samples_seen_, ds_size, n_samples = ups_ds_size[0], ups_ds_size[1], len(inpt)

        if n_samples_seen_ < ds_size and updating > 0.:
            new_weight, new_S, new_explained_var, new_mean_, new_var_, = batch_pca_update(inpt, weight, S, mean_, var_, ups_ds_size, updating, n_components_, n_samples, n_samples_seen_)
            return grad_input, new_weight, new_S, new_explained_var, new_mean_, new_var_, None, None
        else:
            return grad_input, weight.clone(), S.clone(), explained_var.clone(), mean_.clone(), var_.clone(), None, None
  
class Batch_PCA_Layer(nn.Module):

    def __init__(self, n_in, n_out, ds_size=2, updating=True):
        super().__init__()
        if n_in == n_out:
            self.weight = NoGDParameter(torch.eye(n_out))
        else:
            self.weight = NoGDParameter(torch.ones(n_out, n_in)/10)
        self.S = NoGDParameter(torch.zeros(n_out))
        self.var_expl = NoGDParameter(torch.ones(n_out))

        self.mean_ = NoGDParameter(torch.zeros(n_in))
        self.var_ = NoGDParameter(torch.zeros(n_in))

        self.ups_ds_size = torch.Tensor([0., ds_size])
        self.n_components = n_out
        self.updating = torch.ones(1) if updating else torch.zeros(1)

    @property
    def sphering_matrix(self):
        return self.weight.T.detach() / torch.sqrt(self.var_expl).detach()

    @property
    def components(self):
        return self.weight.T.detach()

    @property
    def explained_variance_(self):
        return self.var_expl.detach()
    
    def compute_updates(self, updating=True, ds_size=2):
        self.updating = torch.ones(1) if updating else torch.zeros(1)
        self.ups_ds_size.data[1] = ds_size

    def forward(self, X):
        return F_Batch_PCA.apply(X, self.weight, self.S, self.var_expl, self.mean_, self.var_, self.ups_ds_size, self.updating)

class F_Batch_PCA_2d(Function):

    @staticmethod
    def forward(ctx, inpt, weight, S, explained_var, mean_, var_, bias, ups_ds_size, updating, n_components, filter_size, stride):
        B, C, H, W = inpt.shape
        # rollout the weights
        kernel = weight.view(n_components, C, filter_size, filter_size).clone()
        ctx.save_for_backward(inpt, weight, kernel, stride, S, explained_var, mean_, var_, bias, ups_ds_size, updating) # save unfolded without mean correction for backward
        # begin ugly unfold -> mean correction -> fold
        # inpt = F.unfold(inpt, filter_size, dilation=1, padding=0, stride=(stride, stride)) # (B, C*F*F, n_tiles)
        # inpt = inpt.transpose(1,2).contiguous()
        # inpt = inpt.view(-1, C*filter_size*filter_size)
        # inpt = inpt - mean_ # mean correction
        # inpt = inpt.view(B, -1, C*filter_size*filter_size)
        # inpt = inpt.transpose(1, 2).contiguous()
        # inpt = F.fold(inpt, (H, W), filter_size, dilation=1, padding=0, stride=(stride, stride)) # (B, F*F, n_tiles)
        # end
        outpt = F.conv2d(inpt, kernel, stride=stride.item()) # (B, n_components, H, W)
        outpt =  (outpt - bias.view(1, -1, 1, 1)) / torch.sqrt(explained_var).view(1, -1, 1, 1)
        return outpt

    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, kernel, stride, S, explained_var, mean_, var_, bias, ups_ds_size, updating = ctx.saved_tensors
        B, C, H, W = inpt.shape
        filter_size = kernel.shape[2]
        # Gradient w.r.t. input
        grad_input = F.grad.conv2d_input(grad_output=grad_output, weight=kernel, stride=stride.item(), input_size=inpt.shape)
        
        # Gradient w.r.t. weights # Gradient w.r.t. bias
        n_components_ = weight.shape[0]
        n_samples_seen_, ds_size, n_samples = ups_ds_size[0].clone(), ups_ds_size[1].clone(), len(inpt) # we need .clone() because o.w. new_bias would already get the new value
        T = ((H - filter_size) // stride.item() + 1)**2

        if n_samples_seen_ < ds_size*T and updating > 0.:
            
            inpt_ = F.unfold(inpt, filter_size, dilation=1, padding=0, stride=(stride, stride)) # (B, F*F, n_tiles)
            inpt_ = inpt_.transpose(1,2).contiguous()
            inpt_ = inpt_.view(-1, C*filter_size*filter_size)
            new_weight, new_S, new_explained_var, new_mean_, new_var_ = batch_pca_update(inpt_, weight, S, mean_, var_, ups_ds_size, updating, n_components_, n_samples * T, n_samples_seen_)

            kernel = new_weight.view(n_components_, C, filter_size, filter_size).clone()
            outpt = F.conv2d(inpt, kernel, stride=stride.item()) # (B, n_components, H, W)
            outpt = outpt.transpose(1,2).transpose(2,3).contiguous().view(-1, n_components_)
            
            new_bias, _, _ =  incremental_mean_and_var(
                                                        outpt, last_mean=bias, last_variance=bias,
                                                        last_sample_count=n_samples_seen_)
    
            return grad_input, new_weight, new_S, new_explained_var, new_mean_, new_var_, new_bias, None, None, None, None, None
        else:
            return grad_input, weight.clone(), S.clone(), explained_var.clone(), mean_.clone(), var_.clone(), bias, None, None, None, None, None

class Batch_PCA_Layer2d(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride=1,  ds_size=2, updating=True):
        super().__init__()
        n_in = in_channels*filter_size*filter_size
        n_out = out_channels
        self.filter_size = filter_size
        self.stride = torch.ones(1).int() * stride

        if n_in == n_out:
            self.weight = NoGDParameter(torch.eye(n_out))
        else:
            self.weight = NoGDParameter(torch.ones(n_out, n_in)/10)
        self.S = NoGDParameter(torch.zeros(n_out))
        
        self.var_expl = NoGDParameter(torch.ones(n_out))
        self.bias = NoGDParameter(torch.zeros(n_out))

        self.mean_ = NoGDParameter(torch.zeros(n_in))
        self.var_ = NoGDParameter(torch.zeros(n_in))

        self.ups_ds_size = torch.Tensor([0., ds_size])
        self.n_components = n_out
        self.updating = torch.ones(1) if updating else torch.zeros(1)

    @property
    def sphering_matrix(self):
        return self.weight.T.detach() / torch.sqrt(self.var_expl).detach()

    @property
    def components(self):
        return self.weight.T.detach()

    @property
    def explained_variance_(self):
        return self.var_expl.detach()
    
    def compute_updates(self, updating=True, ds_size=2):
        self.updating = torch.ones(1) if updating else torch.zeros(1)
        self.ups_ds_size.data[1] = ds_size

    def forward(self, X):
        """
        Args:
            X : input tensor with shape (B, C, H, W)
        """
        assert X.ndim == 4
        return F_Batch_PCA_2d().apply(X, self.weight, self.S, self.var_expl, self.mean_, self.var_, self.bias, self.ups_ds_size, self.updating, self.n_components, self.filter_size, self.stride)