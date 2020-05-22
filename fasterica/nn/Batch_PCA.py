import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from ..nn import NoGDParameter

class F_Batch_PCA(Function):
    """
    https://github.com/scikit-learn/scikit-learn/blob/483cd3eaa/sklearn/decomposition/_incremental_pca.py
    """

    @staticmethod
    def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
        """ 
        T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
        """
        last_sum = last_mean * last_sample_count
        new_sum = X.sum(0, keepdim=True)
        new_sample_count = len(X)
        updated_sample_count = last_sample_count + new_sample_count
        updated_mean = (last_sum + new_sum) / updated_sample_count

        if last_variance is None:
            updated_variance = None
        else:
            new_unnormalized_variance = X.var(0, keepdim=True) * new_sample_count
            last_unnormalized_variance = last_variance * last_sample_count
        
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                    last_unnormalized_variance + new_unnormalized_variance +
                    last_over_new_count / updated_sample_count *
                    (last_sum / last_over_new_count - new_sum) ** 2)

            zeros = last_sample_count == 0
            updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
            updated_variance = updated_unnormalized_variance / updated_sample_count
        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def svd_flip(u, v):
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        n_some = u.shape[1]
        u = u * signs[:n_some]
        v = v * signs.unsqueeze(1)
        return u, v

    @staticmethod
    def forward(ctx, inpt, weight, S, explained_var, mean_, var_, ups_ds_size, updating):
        ctx.save_for_backward(inpt, weight, S, explained_var, mean_, var_, ups_ds_size, updating)
        return  inpt.mm(weight.T)  / torch.sqrt(explained_var)
        
    @staticmethod
    def backward(ctx, grad_output):
        inpt, weight, S, explained_var, mean_, var_, ups_ds_size, updating = ctx.saved_tensors
        # Gradient w.r.t. input
        grad_input = grad_output.mm(weight)
        
        # Gradient w.r.t. weights # Gradient w.r.t. bias
        n_components_ = weight.shape[0]
        n_samples_seen_, ds_size, n_samples = ups_ds_size[0], ups_ds_size[1], len(inpt)

        if n_components_ > len(inpt):
            raise ValueError(f"n_components={n_components_} must be less or equal to the batch number of samples {len(inpt)}")

        if n_samples_seen_ <= ds_size and updating > 0.:

            col_mean, col_var, n_total_samples =  F_Batch_PCA._incremental_mean_and_var(
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
            # We need to swith to numpy as torch-svd is really inaccurate in v.0.4.2
            #U, S, V = torch.svd(inpt, some=False) 
            #U, S, V = U.cuda().float(), S.cuda().float(), V.cuda().float()
            device = inpt.device
            U, S, V = np.linalg.svd(inpt.cpu().numpy(), full_matrices=False)
            U, S, V = torch.from_numpy(U).to(device),torch.from_numpy(S).to(device), torch.from_numpy(V).to(device)

            U, V =  F_Batch_PCA.svd_flip(U, V)
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
        self.weight = NoGDParameter(torch.ones(n_out, n_in))
        self.S = NoGDParameter(torch.zeros(n_out))
        self.bias = NoGDParameter(torch.ones(n_out))

        self.mean_ = NoGDParameter(torch.zeros(n_in))
        self.var_ = NoGDParameter(torch.zeros(n_in))

        self.ups_ds_size = torch.from_numpy(np.asarray([0., ds_size])).float()
        self.n_components = n_out
        self.updating = torch.ones(1) if updating else torch.zeros(1)
    
    def compute_updates(self, updating=True, ds_size=2):
        self.updating = torch.ones(1) if updating else torch.zeros(1)
        self.ups_ds_size.data[1] = ds_size

    def forward(self, X):
        return F_Batch_PCA.apply(X, self.weight, self.S, self.bias, self.mean_, self.var_, self.ups_ds_size, self.updating)