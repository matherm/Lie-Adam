import torch
import numpy as np
import scipy

class Loss():

    @staticmethod
    def Logcosh(x, a1=1.2):
        """is proportional negative log-likelihood
        """
        # inverse soft hat
        ax = a1*x
        if ax.max() > 80 or ax.min() < -80:
            warnings.warn(f"Exceeding range of cosh(). Maybe decrease a1: ({a1})")
            ax = torch.clamp(ax, -80, 80)
        return -(-torch.log( torch.cosh( ax ) + 1e-5 )/a1)

    @staticmethod
    def Exp(x, a2=0.99):
        """is proportional negative log-likelihood
        """
        # hat
        return -(torch.exp(-a2/2*x**2)/a2)

    @staticmethod
    def FrobCov(X, W):
        return np.linalg.norm(np.cov((X @ W).T) - np.eye(W.shape[1])) / W.shape[1]

    @staticmethod
    def Kurtosis(X, W):
        return scipy.stats.kurtosis(X @ W).mean()