import torch
import numpy as np
import torch.nn.functional as F

def to_norm_contrast(X, DC=True, channels=None, eps=0.2):
    assert X.min() >= 0 and X.max() <= 1
    if channels is not None:
        X_ = X.reshape(len(X), channels, -1)
        sldj = 0
        X_channels = []
        for c in range(channels):
            X_channel, sldj_channel = to_norm_contrast(X_[:, c], DC, channels=None, eps=eps) 
            X_channels.append(X_channel)
            sldj += sldj_channel
        return np.concatenate(X_channels, axis=1), sldj
    else:
        dim = X.shape[1]
        if DC:
            X = X - X.mean(1, keepdims=True) # DC
        norm = np.clip(np.linalg.norm(X, axis=1, keepdims=True), a_min=eps, a_max=None)
        X_ = X / norm # Contrast
        sldj = -np.log(norm.flatten())*dim
        nans = np.isnan(X_)
        if torch.is_tensor(X_):
            nans = nans.bool()
        X_[nans] = X[nans]
        infs = np.isinf(sldj)
        sldj[infs] = 0
        return X_, sldj

def scale(X, mean, std):
    X = (X - mean) / std
    if type(np.asarray(std)) == np.ndarray:
        sldj = -np.log(std).sum()
    else:
        sldj = -np.log(std)*dim
    return X, sldj

def dequantize(X, bits=256):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X = X + np.random.uniform(0, 1/bits, X.shape).astype(np.float32)
    X = np.clip(X, 0, 1)
    return X, -np.log(bits)*dim

def to_logit(x):
    assert x.min() >= 0 and x.max() <= 1
    if not torch.is_tensor(x):
        y, sldj = to_logit(torch.FloatTensor(x))
        return y.numpy(), sldj.numpy()
    noise = torch.FloatTensor(np.random.uniform(size=x.shape)).to(x.device)
    data_constraint = torch.FloatTensor([0.9]).to(x.device)
    y = (x * 255. + noise) / 256.
    y = (2 * y - 1) * data_constraint
    y = (y + 1) / 2
    y = torch.log(y + 1e-8) - torch.log(1. - y)

    # Save log-determinant of Jacobian of initial transform
    ldj = F.softplus(y) + F.softplus(-y) \
        - F.softplus((1. - torch.log(data_constraint) - torch.log(data_constraint)))
    sldj = ldj.view(ldj.size(0), -1).sum(-1)
    return y, sldj

def contrast_norm_preprocessing(X, In, Out, a, b, c):
    
    # DC
    X, In, Out = X - X.mean((1,2,3), keepdims=True), In - In.mean((1,2,3), keepdims=True), Out - Out.mean((1,2,3), keepdims=True)

    # Contrast
    X, In, Out = X / np.linalg.norm(X.reshape(len(X), -1), axis=1, keepdims=True)[:, :, None, None], In / np.linalg.norm(In.reshape(len(In), -1), axis=1, keepdims=True)[:, :, None, None], Out / np.linalg.norm(Out.reshape(len(Out), -1), axis=1, keepdims=True)[:, :, None, None]

    # Rescale
    scale = X.std()
    X = X / scale
    In = In / scale
    Out = Out / scale    
    
    return X, In, Out, 0, scale