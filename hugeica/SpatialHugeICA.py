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
        self.n_tiles   = ((shape[1]-BSZ[0])//stride+1)**2

    def col2i(self, X, len_X):
        col2i = torch.FloatTensor(
                                    col2im(BSZ=self.BSZ, agg="avg", cols=X.cpu().numpy().T, x_shape=(len_X, *self.shape), padding=self.padding, stride=self.stride)
                                ).to(X.device)
        return col2i
    
    def i2col(self, X):
        i2col = torch.FloatTensor(
                                    im2col(X.reshape(len(X), *self.shape).cpu().numpy(), BSZ=self.BSZ, padding=self.padding, stride=self.stride).T
                                ).to(X.device)

        return i2col

    
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
        X_ = self.i2col(X)
        n_tiles   = len(X_) // len(X)
        n_images  = len(X_) // n_tiles
        n_patches = len(X_)
        # see: https://stackoverflow.com/questions/59520967/super-keyword-doesnt-work-properly-in-list-comprehension
        sup_transform = super().transform 
        s = torch.cat([sup_transform(X_[i:i+self.bs]) for i in range(0, len(X_), self.bs)], dim=0) # transform the patches
        s = torch.pow(s, exponent=exponent)
        s = act(s)
        s = s[im2colOrder(n_images, n_patches)]         # reorder such that patches of same image are consecutive
        s = s.reshape(-1, n_tiles, n_components)
        if resample == "with_replacement":
            # This breaks the temporal dependencies and takes a random subset of patches
            s = torch.stack([ s[i, torch.randint(n_tiles, (n_tiles,)), :] for i in range(len(s)) ])
        if resample == "without_replacement":
            # This breaks the temporal dependencies and takes a random subset of patches
            s = torch.stack([ s[i, torch.randperm(n_tiles), :] for i in range(len(s)) ])
        s = self.agg(s, agg)
        return s

    def set_residuals_std(self, X):
        """
        Computes the residuals for the ELBO estimation.
        """
        if X.shape[1] > self.d:
            X = self.i2col(X)
        super().set_residuals_std(X)

    def agg(self, s, agg):
        if agg == "mean":
            s = s.mean(1) # average over patches of the same image
        elif agg == "diff":
            s = sum_of_diff(s, 1) # average over patches of the same image
        elif agg == "coherence":
            s = s
            st = torch.roll(s, 1, dims=1)
            s = (s * st).sum(2, keepdims=True) # dot product between components
            s = s.mean(1) # temporal average
        elif agg == "shufflediff":
            idx = np.random.permutation(np.arange(n_tiles))
            s = sum_of_diff(s[:, idx, :], 1) # average over patches of the same image
        elif agg == "sum":
            s = s.sum(1) # sum over patches of the same image
        elif agg == "shufflesum":
            idx = np.random.permutation(np.arange(n_tiles))
            s = s[:, idx, :].sum(1) # sum over patches of the same image
        elif agg == "std":
            s = s.std(1) # sum over patches of the same image
        elif agg == "var":
            s = s.var(1) # sum over patches of the same image
        elif agg == "prod":
            s = s.prod(1) # sum over patches of the same image
        elif agg == "max":
            s = s.max(1)[0] # sum over patches of the same image
        elif agg == "min":
            s = s.min(1)[0] # sum over patches of the same image
        elif agg == "skew":
            s = s - s.mean(1, keepdims=True)
            s = s - s.std(1, keepdims=True)
            s = (s**3).mean(1)
        elif agg == "invprod": 
            s = torch.reciprocal(s).prod(1) # sum over patches of the same image
        elif agg == "invsum":
            s = torch.reciprocal(s).sum(1) # sum over patches of the same image
        elif agg == "none":
            s = s
        else:
            raise ValueError(f"agg == {agg} not understood.") 
        return s
    
    def score_norm(self, X, ord=2, exponent=1):
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
        patches, z, z_ = self.predict(X)
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
            H_qz_q = entropy_gaussian(dim=k*n_tiles)
            H_qz_q = -torch.distributions.Normal(z.flatten(), 1).log_prob(z_.flatten()).reshape(n, -1).sum(1)
        else:
            X_, z, z_ = self.predict(X, sample_scale=sample_scale)
            z = z[im2colOrder(n_images, len(z))]
            z_ = z
            H_qz_q = np.zeros(n)

        X, X_, z, z_ = X.cpu(), X_.cpu(), z.cpu(), z_.cpu()

        #sigma_per_dim = self.sigma_residuals.repeat(len(X)) + sigma_eps # add minimal variance
        #sigma_per_dim = torch.ones_like(X.flatten()) # add minimal variance
        #print(X.shape, "distributions")
        #log_px_z = torch.distributions.Normal(X.flatten(), sigma_per_dim)
        #print(X.shape, "distributions", X_.flatten().mean())
        #log_px_z = log_px_z.log_prob(X_.flatten()).reshape(len(X), -1)
        #print(X.shape, "col2i")
        
        # log_px_z = 0.5*(-np.log(2*np.pi) - ((X.flatten() - X_.flatten())**2)).reshape(len(X), -1)
        # log_px_z = self.col2i(log_px_z, n)
        # log_px_z = log_px_z.reshape(n, -1).sum(1)
        images = self.col2i(X, n) 
        images_rec = self.col2i(X_, n)
        log_px_z = 0.5*(-np.log(2*np.pi) - ((images.flatten() - images_rec.flatten())**2))
        log_px_z = log_px_z.reshape(len(images), -1).sum(1) # sum over pixels

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

        images = self.col2i(X, n) 
        images_rec = self.col2i(X_, n)
        log_px_z = 0.5*(-np.log(2*np.pi) - ((images.flatten() - images_rec.flatten())**2))
        log_px_z = log_px_z.reshape(len(images), -1).sum(1) # sum over pixels
        
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

        """
        if not torch.is_tensor(X):
            return self.forward(torch.FloatTensor(X))
        if X.dim() == 2:
            X_ = self.i2col(X)
            X_ = X_[im2colOrder(len(X), len(X_))] # reordering
        return super().transform(X_)
    
    def fit(self, X, epochs, X_val, lr=1e-3, resample=False, *args, **kwargs):
        if not torch.is_tensor(X):
            return self.fit(torch.FloatTensor(X), epochs, torch.FloatTensor(X_val), resample, *args, **kwargs)

        if self.net is None:
            X_ = self.i2col(X)         # patching
            X_val_ = self.i2col(X_val) # patching
            X_ = X_[im2colOrder(len(X), len(X_))] # reordering
            X_val_ = X_val_[im2colOrder(len(X_val), len(X_val_))] # reordering
        else:
            X_ = X
            X_val_ = X_val
            
        if resample == "with_replacement":
            # This breaks the temporal dependencies and takes a random subset of patches
            X_ = X_.view(len(X), self.n_tiles, -1)
            X_ = torch.stack([ X_[i, torch.randint(self.n_tiles, (self.n_tiles,)), :] for i in range(len(X)) ])
            
            X_val_ = X_val_.view(len(X_val_), self.n_tiles, -1)
            X_val_ = torch.stack([ X_val_[i, torch.randint(self.n_tiles, (self.n_tiles,)), :] for i in range(len(X_val_)) ])

        super().fit(X_, epochs, X_val_, *args, **kwargs)
