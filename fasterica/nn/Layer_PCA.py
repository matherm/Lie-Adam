import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from ..nn import NoGDParameter
from ..helpers import *

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


        if n_samples_seen_ <= ds_size and updating > 0.:

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
            U, S, V = torch.svd(inpt, some=True) # svd returns V instead of numpy-ish V.T
            U, S, V = U.to(device), S.to(device), V.to(device) 
            U, V =  svd_flip(U, V.T)
            explained_variance = S ** 2 / (n_total_samples - 1) 
    
            ups_ds_size.data[0] = n_total_samples
            new_weight = V[:n_components_]
            new_S = S[:n_components_]
            new_explained_var = explained_variance[:n_components_]
            new_mean_ = col_mean
            new_var_ = col_var
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
        self.bias = NoGDParameter(torch.ones(n_out))

        self.mean_ = NoGDParameter(torch.zeros(n_in))
        self.var_ = NoGDParameter(torch.zeros(n_in))

        self.ups_ds_size = torch.Tensor([0., ds_size])
        self.n_components = n_out
        self.updating = torch.ones(1) if updating else torch.zeros(1)

    @property
    def sphering_matrix(self):
        return self.weight.T.detach() / torch.sqrt(self.bias).detach()

    @property
    def explained_variance_(self):
        return self.bias.detach()
    
    def compute_updates(self, updating=True, ds_size=2):
        self.updating = torch.ones(1) if updating else torch.zeros(1)
        self.ups_ds_size.data[1] = ds_size

    def forward(self, X):
        return F_Batch_PCA.apply(X, self.weight, self.S, self.bias, self.mean_, self.var_, self.ups_ds_size, self.updating)