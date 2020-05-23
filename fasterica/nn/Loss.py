import torch, warnings, scipy
import numpy as np
from ..helpers.mutual_information import mutual_information

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
    def FrobCov(S):
        return np.linalg.norm(np.cov((S).T) - np.eye(S.shape[1])) / S.shape[1]

    @staticmethod
    def Kurtosis(S):
        return scipy.stats.kurtosis(S).mean()

    @staticmethod
    def MI(S):
        return mutual_information(S)