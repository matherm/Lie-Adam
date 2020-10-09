import torch
import numpy as np
import torch.nn.functional as F

def to_norm_contrast(X):
    assert X.min() >= 0 and X.max() <= 1
    dim = X.shape[1]
    X = X - X.mean(1, keepdims=True) # DC
    X = X / np.linalg.norm(X, axis=1, keepdims=True) # Contrast
    sldj = -np.log(np.linalg.norm(X, axis=1).flatten())*dim
    return X, sldj

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