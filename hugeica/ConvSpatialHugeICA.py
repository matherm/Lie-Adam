import torch, warnings, time, sys
import torch.nn as nn
import numpy as np
import itertools
import scipy
from tqdm import tqdm
from hugeica import *

class Net2d(nn.Module):

    def __init__(self, inpt_shape, out_channels, filter_size, stride, ds_size, whiten, ica):
        super().__init__()

        self.inpt = Reshape((-1, *inpt_shape))
        self.whiten = Batch_PCA_Layer2d(inpt_shape[0], out_channels, filter_size=filter_size, stride=stride,  ds_size=ds_size, updating=True)
        self.ica    = SO_Layer2d(out_channels, filter_size=1, stride=1)
        self.transpose = nn.Sequential(Transpose(1, 2), Transpose(2, 3))
        self.output = Reshape((-1, out_channels))
        if not ica:
            self.ica.weight.data = torch.eye(self.ica.weight.data.shape[0])
        self.net = nn.Sequential(self.whiten, self.ica)
        
    def transform(self, X):
        return self.transpose(self.net(self.inpt(X)))

    def forward(self, X):
        return self.output(self.transpose(self.net(self.inpt(X))))

class ConvSpatialICA(SpatialICA):
    

    def transform(self, X, exponent=1, agg="mean", act=lambda x : x, resample=False):
        """
        Transforms the data matrix X spatially.

        Args:
            X (B, d) : The data matrix.
            exponent (int) : The exponent of the activation function s = (XWR)^{exponent}
            agg (String) : The aggrecation function ('mean', 'sum', 'std', 'var' 'prod' or 'none')

        Returns
            s (B, k) : Spatially averaged components s 
        """ 
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.transform(torch.FloatTensor(X).to(device), exponent=exponent, agg=agg, act=act, resample=resample).cpu().detach().numpy()
        n_components = self.n_components
        device = next(self.parameters()).device
        s = torch.cat([self.net.transform(X[i:i+self.bs].to(device)) for i in range(0, len(X), self.bs)], dim=0) # transform the patches
        s = torch.pow(s, exponent=exponent)
        s = act(s)
        s = s.reshape(len(s), -1, n_components)
        if resample == "with_replacement":
            # This breaks the temporal dependencies and takes a random subset of patches
            s = torch.stack([ s[i, torch.randint(n_tiles, (n_tiles,)), :] for i in range(len(s)) ])
        if resample == "without_replacement":
            # This breaks the temporal dependencies and takes a random subset of patches
            s = torch.stack([ s[i, torch.randperm(n_tiles), :] for i in range(len(s)) ])
        s = self.agg(s, agg)
        return s

    def fit(self, X, epochs, X_val, lr=1e-3, *args, **kwargs):
        if not torch.is_tensor(X):
            return self.fit(torch.FloatTensor(X), epochs, torch.FloatTensor(X_val), *args, **kwargs)

        if self.net is None:
            self.d = self.shape[0]*self.BSZ[0]**2
            self.net = Net2d(self.shape, self.n_components, filter_size=self.BSZ[0], stride=self.stride,  ds_size=int(len(X) * self.optimistic_whitening_rate), whiten=self.whiten, ica=self.ica)
            self.reset(lr)

        super().fit(X, epochs, X_val, *args, **kwargs)