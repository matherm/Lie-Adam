import numpy as np
from ..nn.Loss import Loss
from ..data.Preprocessing import *
from .im2col import *

def bpd_pca_logit(model, X, mean, std):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.log_prob(X) + sldj_inter + sldj_logit + sldj_scale
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_pca_normal(model, X, mean, std):
    """
    # log determinant of jacobean
    # [0 , 255] # -128 --> [-128, 128] --> 0
    # [-128, 128] # /128 --> [-1, 1] --> np.log(1/128)*dim
    # np.log(1/128)*dim = dim*(log(1) - log(128)) = -dim*log(128)

    # p(x) = p(f^-1(x))|det df f^-1(x)|
    # log p(x) = log p(z) + log_det_J # Division is minus
    """
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_contr = to_norm_contrast(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.log_prob(X) + sldj_inter + sldj_scale + sldj_contr
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X


def bpd_pca_elbo(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_contr = to_norm_contrast(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X, p_z) + sldj_inter + sldj_scale + sldj_contr
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_pca_elbo_receptive(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    n = X.shape[0]
    X = model.i2col(torch.FloatTensor(X))
    X = X[im2colOrder(n, len(X))]
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_contr = to_norm_contrast(X)
    X, sldj_scale = scale(X, np.asarray(mean).mean(), np.asarray(std).mean())
    elbo_X = super(type(model), model).elbo(X, p_z) + sldj_inter + sldj_scale + sldj_contr
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X


def bpd_pca_elbo_normal(model, X, mean, std):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_contr = to_norm_contrast(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X, Loss.Gaussian) + sldj_inter + sldj_scale + sldj_contr
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_pca_elbo_logit(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X, p_z) + sldj_inter + sldj_logit + sldj_scale
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def R_logit(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    cex = model.cross_entropy_x(X, p_z)
    return cex


def H_logit(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    H = model.entropy(X, p_z)
    return H
    
def CE_logit(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    ce = model.cross_entropy(X, p_z)
    return ce

def bits_back_code(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    bb = model.bits_back_code(X, p_z)
    return bb

def bits_back_code_per_dim(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    bb = model.bits_back_code(X, p_z, per_dim=True)
    return bb

def bpd_elbo_logit(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X, p_z) + sldj_inter + sldj_logit + sldj_scale
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_elbo(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_contr = to_norm_contrast(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X, p_z) + sldj_inter + sldj_contr + sldj_scale
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_elbo_logit_receptive(model, X, mean, std, p_z=Loss.Gaussian):
    assert X.min() >= 0 and X.max() <= 1
    n = X.shape[0]
    X = model.i2col(torch.FloatTensor(X))
    X = X[im2colOrder(n, len(X))]
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = super(type(model), model).elbo(X, p_z) + sldj_inter + sldj_logit + sldj_scale
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_ica_normal(model, X, mean, std):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_contr = to_norm_contrast(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X) + sldj_inter + sldj_scale + sldj_contr
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X

def bpd_ica_logit(model, X, mean, std):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X, sldj_inter = dequantize(X)
    X, sldj_logit = to_logit(X)
    X, sldj_scale = scale(X, mean, std)
    elbo_X = model.elbo(X) + sldj_inter + sldj_logit + sldj_scale
    bpd_X = -elbo_X/(np.log(2) * dim)
    return bpd_X