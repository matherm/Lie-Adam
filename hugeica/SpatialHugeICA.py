import torch, warnings, time, sys
import torch.nn as nn
import numpy as np
import itertools
import scipy
from tqdm import tqdm
from hugeica import *


class SpatialICA(HugeICA):
    
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
    
    def transform(self, X, exponent=1, agg="mean", bs=-1, act=lambda x : x):
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
            return self.transform(torch.FloatTensor(X).to(device), exponent=exponent, agg=agg, bs=bs, act=act).cpu().detach().numpy()
        X_ = self.i2col(X)
        n_tiles   = len(X_) // len(X)
        n_images  = len(X_) // n_tiles
        n_patches = len(X_)
        n_components = self.n_components
        if bs < 0:
            bs = len(X)
        # see: https://stackoverflow.com/questions/59520967/super-keyword-doesnt-work-properly-in-list-comprehension
        sup_transform = super().transform 
        s = torch.cat([sup_transform(X_[i:i+bs]) for i in range(0, len(X_), bs)], dim=0) # transform the patches
        s = torch.pow(s, exponent=exponent)
        s = act(s)
        s = s[im2colOrder(n_images, n_patches)]         # reorder such that patches of same image are consecutive
        if agg == "mean":
            s = s.reshape(-1, n_tiles, n_components).mean(1) # average over patches of the same image
        elif agg == "diff":
            s = sum_of_diff(s.reshape(-1, n_tiles, n_components),1) # average over patches of the same image
        elif agg == "sum":
            s = s.reshape(-1, n_tiles, n_components).sum(1) # sum over patches of the same image
        elif agg == "std":
            s = s.reshape(-1, n_tiles, n_components).std(1) # sum over patches of the same image
        elif agg == "var":
            s = s.reshape(-1, n_tiles, n_components).var(1) # sum over patches of the same image
        elif agg == "prod":
            s = s.reshape(-1, n_tiles, n_components).prod(1) # sum over patches of the same image
        elif agg == "max":
            s = s.reshape(-1, n_tiles, n_components).max(1)[0] # sum over patches of the same image
        elif agg == "min":
            s = s.reshape(-1, n_tiles, n_components).min(1)[0] # sum over patches of the same image
        elif agg == "invprod": 
            s = torch.reciprocal(s).reshape(-1, n_tiles, n_components).prod(1) # sum over patches of the same image
        elif agg == "invsum":
            s = torch.reciprocal(s).reshape(-1, n_tiles, n_components).sum(1) # sum over patches of the same image
        elif agg == "none":
            s = s.reshape(-1, n_tiles, n_components)
        else:
            raise ValueError(f"agg == {agg} not understood.") 
        return s
    
    def score_norm(self, X, ord=0.5, exponent=1):
        """
        Computes the norm of the transformed components.
        ||E_patches[s]||
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.score_norm(torch.FloatTensor(X).to(device), ord=ord, exponent=exponent).cpu().detach().numpy()
        s = self.transform(X, exponent=exponent)
        return torch.norm(s, dim=1, p=ord)
    
    def score(self, X):
        """
        Computes the loss for the given samples.
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.score(torch.FloatTensor(X).to(device)).cpu().detach().numpy()
        s = self.transform(X)
        return self.loss(s).mean(1)

    def predict(self, X, sample_scale=0.):
        """
        Compresses the input.
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            X_, z, z_ = self.predict(torch.FloatTensor(X).to(device), sample_scale)
            return X_.cpu().numpy(), z.cpu().numpy(), z_.cpu().numpy()
        if X.shape[1] > self.d:
            X = self.i2col(X)
        return super().predict(X, sample_scale)

    def reconstruct(self, X):
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.reconstruct(torch.FloatTensor(X).to(device)).cpu().detach().numpy()
        patches, z = self.predict(X)
        X_ = self.col2i(patches, len(X)).reshape(len(X), -1)
        return X_

   
    def elbo(self, X, p_z=Loss.ExpNormalized, sigma_eps=1e-5,  sample_scale=0.):
        """
        Elbo computation for spatial ICA. See ``HugeICA.elbo()`` for the derivation.
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.elbo(torch.FloatTensor(X).to(device), p_z, sigma_eps).cpu().detach().numpy()
        n, d, k = len(X), self.d, self.n_components
        n_tiles = len(X) // n
        n_images = len(X)

        X = self.i2col(X)
        if sample_scale > 0.:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z_[im2colOrder(n_images, len(z))]
            H_qz_q = entropy_gaussian(np.eye(k*n_tiles))
            H_qz_q = -torch.distributions.Normal(z.flatten(), 1).log_prob(z_.flatten()).reshape(n, -1).sum(1)
        else:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z
            H_qz_q = np.zeros(n)

        X, X_, z, z_ = X.cpu(), X_.cpu(), z.cpu(), z_.cpu()

        sigma_per_dim = self.sigma_residuals.repeat(len(X)) + sigma_eps # add minimal variance
        log_px_z = torch.distributions.Normal(X.flatten(), sigma_per_dim).log_prob(X_.flatten()).reshape(len(X), -1)
        log_px_z = self.col2i(log_px_z, n)
        log_px_z = log_px_z.reshape(n, -1).sum(1)

        log_pz_z = p_z(z_)
        log_pz_z = log_pz_z.reshape(n, -1).sum(1) # sum over timesteps
        
        return log_px_z + log_pz_z + H_qz_q
    
    def entropy(self, X, p_z=Loss.ExpNormalized, sigma_eps=1e-5,  sample_scale=0.):
        """
        H(Q(Z))
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.entropy(torch.FloatTensor(X).to(device), p_z, sigma_eps).cpu().detach().numpy()
        n, d, k = len(X), self.d, self.n_components
        n_tiles = len(X) // n
        n_images = len(X)

        X = self.i2col(X)
        if sample_scale > 0.:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z_[im2colOrder(n_images, len(z))]
            H_qz_q = entropy_gaussian(np.eye(k*n_tiles))
            H_qz_q = -torch.distributions.Normal(z.flatten(), 1).log_prob(z_.flatten()).reshape(n, -1).sum(1)
        else:
            H_qz_q = np.zeros(n)

        return H_qz_q
    
    def cross_entropy(self, X, p_z=Loss.ExpNormalized, sigma_eps=1e-5,  sample_scale=0.):
        """
        CE(p(Z)|Q(Z|X))
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.cross_entropy(torch.FloatTensor(X).to(device), p_z, sigma_eps, sample_scale).cpu().detach().numpy()
        n, d, k = len(X), self.d, self.n_components
        n_tiles = len(X) // n
        n_images = len(X)

        X = self.i2col(X)
        if sample_scale > 0.:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z_[im2colOrder(n_images, len(z))]
        else:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z

        X, X_, z, z_ = X.cpu(), X_.cpu(), z.cpu(), z_.cpu()

        log_pz_z = p_z(z_)
        log_pz_z = log_pz_z.reshape(n, -1).sum(1) # sum over timesteps
        #print(n, -(log_pz_z.reshape(n, -1).sum(1).mean()))
        
        return -log_pz_z

    def cross_entropy_x(self, X, p_z=Loss.ExpNormalized, sigma_eps=1e-5,  sample_scale=0.):
        """
        CE(P(X)|P(X|Z))
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.cross_entropy_x(torch.FloatTensor(X).to(device), p_z, sigma_eps).cpu().detach().numpy()
        n, d, k = len(X), self.d, self.n_components
        n_tiles = len(X) // n
        n_images = len(X)

        X = self.i2col(X)
        if sample_scale > 0.:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
        else:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)

        X, X_, z = X.cpu(), X_.cpu(), z.cpu()

        sigma_per_dim = self.sigma_residuals.repeat(len(X)) + sigma_eps # add minimal variance
        log_px_z = torch.distributions.Normal(X.flatten(), sigma_per_dim).log_prob(X_.flatten()).reshape(len(X), -1)
        log_px_z = self.col2i(log_px_z, n)
        log_px_z = log_px_z.reshape(n, -1).sum(1)
        
        return -log_px_z

    def bits_back_code(self, X, p_z=Loss.ExpNormalized, sigma_eps=1e-5, per_dim=False,  sample_scale=0.):
        """
        Bits back code length in bits.
        https://mathoverflow.net/questions/206579/how-do-you-use-the-bits-you-get-back-from-bits-back-coding
        http://users.ics.aalto.fi/harri/thesis/valpola_thesis/node30.html
        $ −log_2(P(z)/Q(z|x)) $
        """
        if not torch.is_tensor(X):
            device = next(self.parameters()).device
            return self.bits_back_code(torch.FloatTensor(X).to(device), p_z, sigma_eps, per_dim).cpu().detach().numpy()
        n, d, k = len(X), self.d, self.n_components
        n_tiles = len(X) // n
        n_images = len(X)

        X = self.i2col(X)
        if sample_scale > 0.:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z_[im2colOrder(n_images, len(z))]
            log_qz_x = torch.distributions.Normal(z.flatten(), 1).log_prob(z_.flatten()).reshape(n, -1).sum(1)
        else:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z
            log_qz_x = np.zeros(n)

        X, X_, z, z_ = X.cpu(), X_.cpu(), z.cpu(), z_.cpu()

        log_pz_z = p_z(z_)
        log_pz_z = log_pz_z.reshape(n, -1).sum(1) # sum over timesteps

        bbc = -(log_pz_z - log_qz_x) / np.log(2) # H[p(z)] - H[q(z)]
        if per_dim:
            return bbc/ (n_tiles * k)
        else:
            return bbc

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
        X_ = X_[im2colOrder(len(X), len(X_))] # reordering
        X_val_ = X_val_[im2colOrder(len(X_val), len(X_val_))] # reordering
        self.n_tiles   = len(X_) // len(X)
        super().fit(X_, epochs, X_val_, *args, **kwargs)