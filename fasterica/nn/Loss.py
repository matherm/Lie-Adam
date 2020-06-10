import torch, warnings, scipy
import numpy as np
from torch.nn.functional import relu
from ..helpers.mutual_information import mutual_information

class Loss():

    @staticmethod
    def NNICA(x):
        """Non-negative ICA
        """
        return 1/2 * torch.pow(-relu(-x), 2)

    @staticmethod
    def Logcosh(x, a1=1.2):
        """is proportional negative log-likelihood
        """
        # inverse soft hat
        ax = a1*x
        if ax.max() > 80 or ax.min() < -80:
            warnings.warn(f"Exceeding range of cosh(). Maybe decreasing a1=({a1}) will help.")
            ax = torch.clamp(ax, -80, 80)
        return -(-torch.log( torch.cosh( ax ) + 1e-5 )/a1)

    @staticmethod
    def Exp(x, a2=0.99):
        """is proportional negative log-likelihood
        """
        # hat
        return -(torch.exp(-a2/2*x**2)/a2)

    @staticmethod
    def LogcoshNormalized(x):
        """is proportional negative log-likelihood
        """
        # inverse soft hat
        return -(-2*torch.log(torch.cosh(np.pi/(2*np.sqrt(3))*x)) - 4*np.sqrt(3)/(np.pi))

    @staticmethod
    def ExpNormalized(x):
        """is proportional negative log-likelihood
        """
        # hat
        return -(torch.exp(-np.sqrt(2)*torch.abs(x))/np.sqrt(2))

    @staticmethod
    def FrobCov(S):
        return np.linalg.norm(np.cov((S).T) - np.eye(S.shape[1])) / S.shape[1]

    @staticmethod
    def Kurtosis(S):
        return scipy.stats.kurtosis(S).mean()

    @staticmethod
    def MI(S):
        return mutual_information(S)

    @staticmethod
    def MI_negentropy(S, G_fun=None, y=np.random.normal(0,1,1000)):
        """
        https://ieeexplore.ieee.org/abstract/document/5226546
        """
        if G_fun is None:
            G_fun = lambda x : Loss.Logcosh(torch.from_numpy(x)).numpy()
            
        E_G_z = G_fun(S).mean(0) 
        E_G_g = G_fun(y).mean(0) 
        
        J_z = (E_G_z - E_G_g)**2
        return -J_z.sum()

    @staticmethod
    def grad_norm(params_old, params_new):
        grad_old = []
        for w in params_old:
            grad_old.append(w.grad.flatten().detach())
        grad_new = []
        for w in params_old:
            grad_new.append(w.grad.flatten().detach())
        return torch.norm(grad_old - grad_new)