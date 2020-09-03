import torch, warnings, time, sys
import torch.nn as nn
import numpy as np
import itertools
import scipy
from tqdm import tqdm
from fasterica import *

class SpatialICA(FasterICA):
    
    def __init__(self, shape, BSZ, stride=4, padding=0, *args, **kwargs):
        """
        Spatial ICA

        Args:
            shape (C, H, W) : The shape of the images
            BSZ (patch_h, patch_w) : the shape of the patches
            padding (float) : 

        Returns
            s (B, k) : Spatially averaged components s 
        """
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.BSZ = BSZ
        self.stride = stride
        self.padding = padding
        
        self.i2col = lambda X: torch.FloatTensor(
                                    im2col(X.reshape(len(X), *shape).cpu().numpy(), BSZ=self.BSZ, padding=self.padding, stride=self.stride).T
                                ).to(X.device)

        self.col2i = lambda X,len_X: torch.FloatTensor(
                                    col2im(BSZ=self.BSZ, agg="avg", cols=X.cpu().numpy().T, x_shape=(len_X, *shape), padding=self.padding, stride=self.stride)
                                ).to(X.device)
    
    def transform(self, X):
        """
        Transforms the data matrix X spatially.

        Args:
            X (B, d) : The data matrix.

        Returns
            s (B, k) : Spatially averaged components s 
        """
        if not torch.is_tensor(X):
            return self.transform(torch.FloatTensor(X))
        X_ = self.i2col(X)
        n_tiles   = len(X_) // len(X)
        n_images  = len(X_) // n_tiles
        n_patches = len(X_)
        n_components = self.n_components
        s = super().transform(X_) # transform the patches
        s = s[im2colOrder(n_images, n_patches)]         # reorder such that patches of same image are consecutive
        s = s.reshape(-1, n_tiles, n_components).mean(1) # average over patches of the same image
        return s
    
    def score_norm(self, X, ord=0.5):
        """
        Computes the norm of the transformed components.
        ||E_patches[s]||
        """
        s = self.transform(X)
        return torch.norm(s, dim=1, p=ord)
    
    def score(self, X):
        """
        Computes the loss for the given samples.
        """
        s = self.transform(X)
        return self.loss(s).mean(1)

    def predict(self, X):
        """
        Compresses the input.
        """
        if not torch.is_tensor(X):
            X_, z = self.predict(torch.FloatTensor(X))
            return X_.cpu().numpy(), z.cpu().numpy()
        if X.shape[1] > self.d:
            X = self.i2col(X)
        return super().predict(X)

    def elbo(self, X):
        """
        Elbo computation for spatial ICA. See ``FasterICA.elbo()`` for the derivation.
        """
        if not torch.is_tensor(X):
            return self.elbo(torch.from_numpy(X)).cpu().detach().numpy()
        n, d, k = len(X), self.d, self.n_components

        X = self.i2col(X)
        X_, z = super().predict(X)
        assert len(X_) == len(X)
        log_px_z = torch.distributions.Normal(X.flatten(), self.sigma_residuals.repeat(len(X))).log_prob(X_.flatten()).reshape(len(X), -1)
        log_px_z = self.col2i(log_px_z, n)
        log_px_z = log_px_z.reshape(n, -1).sum(1)

        log_pz_z = Loss.ExpNormalized(z)[im2colOrder(len(X), len(z))]
        log_pz_z = log_pz_z.reshape(n, -1).sum(1) 
        
        H_qz_q = entropy_gaussian(np.eye(k))
        elbo = log_px_z + log_pz_z + H_qz_q
        return elbo

    def bpd(self, X):
        return -self.elbo(X) / (np.log(2) * self.d)
        
    def forward(self, X):
        """
        Computes the forward pass of data matrix X spatially.
        Args:
            X (B, d) : The data matrix.

        Returns
            S (B*n_tiles, k) : The k components of the patches. 

        Note:
            The returned components patches are not stored consecutive. 
            S = S[im2colOrder(n_images, n_patches)]
            For consecutive order.
        """
        if not torch.is_tensor(X):
            return self.forward(torch.FloatTensor(X))
        X_ = self.i2col(X)
        return super().transform(X_)
    
    def fit(self, X, epochs, X_val, *args, **kwargs):
        if not torch.is_tensor(X):
            return self.fit(torch.FloatTensor(X), epochs, torch.FloatTensor(X_val), *args, **kwargs)
        X_ = self.i2col(X)         # patching
        X_val_ = self.i2col(X_val) # patching
        super().fit(X_, epochs, X_val_, *args, **kwargs)