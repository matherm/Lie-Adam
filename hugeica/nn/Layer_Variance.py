import torch
import torch.nn as nn
import warnings
import numpy as np

class RollingVariance(nn.Module):

    """
    Welford's algorithm
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
    https://stackoverflow.com/questions/5147378/rolling-variance-algorithm

    """

    def __init__(self, n_dim, window_size=2, whiten=True):
        super().__init__()
        self.whiten = whiten

        self.window = nn.Parameter(torch.zeros((window_size, n_dim)), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(n_dim), requires_grad=False)
        self.var_sum = nn.Parameter(torch.zeros(n_dim), requires_grad=False)
        self.index = 0
    
    @property
    def variance(self):
        return self.var_sum.data/len(self.window)
        
    @property
    def std(self):  
        return torch.sqrt(self.variance)

    def update(self, X):
        for x in X:
            next_index = (self.index + 1) % len(self.window)  # oldest x value is at next_index, wrapping if necessary.
            new_mean = self.mean + (x-self.window[next_index]) / len(self.window)
        
            self.var_sum.data = self.var_sum + (x- self.mean) * (x- new_mean) - (self.window[next_index] - self.mean) * (self.window[next_index] - new_mean)
            self.mean.data = new_mean
            self.window.data[next_index] = x.detach()
            self.index = next_index

    def forward(self, X):
        if self.training:
            self.update(X)
        
        if self.whiten:
            return (X - self.mean.data) / self.std
        else:
            return X


class OnlineVariance(nn.Module):

    """
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, n_dim, ds_size=-1, whiten=True):
        super().__init__()
        self.n = 0
        self.mean = nn.Parameter(torch.zeros(n_dim), requires_grad=False)
        self.var_sum = nn.Parameter(torch.zeros(n_dim), requires_grad=False)
        self.whiten = whiten
        self.ds_size = ds_size

    @property
    def variance(self):
        return self.var_sum.data/self.n

    @property
    def std(self):  
        return torch.sqrt(self.variance)
  
    def update(self, X):
        if self.n >= self.ds_size:
            self.n = 0
            self.mean.data = self.mean.data * 0.
            self.var_sum.data = self.var_sum.data * 0.
        n_a = self.n
        n_b = len(X)
        n_ab = n_a + n_b
        delta = X.mean(0) - self.mean

        var_sum = ((X - X.mean(0))**2).sum(0)
        self.var_sum.data = self.var_sum + var_sum + delta**2 * (n_a*n_b)/n_ab 
        self.mean.data = self.mean  + delta * n_b/n_ab
        self.n = n_ab 

    def forward(self, X):
        if self.ds_size < 0:
            warnings.warn("ds_size not set. Causes issues when training more than 1 epoch.")
            self.ds_size = np.iinfo(np.int32).max

        if self.training:
            self.update(X)
        
        if self.whiten:
            return (X - self.mean.data) / self.std
        else:
            return X