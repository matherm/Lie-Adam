import torch, warnings, scipy
import numpy as np
from torch.nn.functional import relu
from ..helpers.mutual_information import mutual_information


class Loss():

    GAMMA = torch.randn(2000).view(-1,1)

    @staticmethod
    def NNICA(x):
        """Non-negative ICA
        """
        return 1/2 * torch.pow(-relu(-x), 2)

    @staticmethod
    def Identity(x):
        """Used for relative gradients
        """
        return x  

    @staticmethod
    def Logcosh(x, a1=1.2):
        """is proportional log-probability
        """
        # inverse soft hat
        ax = a1*x
        if ax.max() > 80 or ax.min() < -80:
            warnings.warn(f"Exceeding range of cosh(). Maybe decreasing a1=({a1}) will help.")
            ax = torch.clamp(ax, -80, 80)

        return torch.log( torch.cosh( ax ) + 1e-5 )/a1

    
    @staticmethod
    def K(x):
        """computes the sign of the Kurtosis
        """
        return  (((x - x.mean(0))/x.std(0))**4).mean(0, keepdims=True)   - 3 

    @staticmethod
    def Exp(x, a2=0.99):
        """is proportional log-probability
        """
        # hat 
        return -torch.exp(-a2/2*x**2)/a2

    @staticmethod
    def Tanh(x):
        return torch.tanh(x)

    @staticmethod
    def LogcoshNormalized(x):
        """is log-probability
        """
        return 2*torch.log(torch.cosh(np.pi/(2*np.sqrt(3))*x)) - 4*np.sqrt(3)/(np.pi)

    @staticmethod
    def ExpNormalized(x):
        """is log-probability
        """
        # hat
        return -torch.exp(-np.sqrt(2)*torch.abs(x))/np.sqrt(2)

    @staticmethod
    def FrobCov(S):
        return np.linalg.norm(np.cov((S).T) - np.eye(S.shape[1])) / S.shape[1]

    @staticmethod
    def Kurtosis(S):
        return scipy.stats.kurtosis(S, axis=0)

    @staticmethod
    def MI(S):
        return mutual_information(S)

    @staticmethod
    def Logprob(S, G_fun=None):
        return G_fun(S).mean(1).mean(0)

    @staticmethod
    def NegentropyLoss(S, G_fun):
        """
        https://ieeexplore.ieee.org/abstract/document/5226546
        """
        
        Loss.GAMMA = Loss.GAMMA.to(S.device)     
        E_G_z = G_fun(S).mean(0) 
        E_G_g = G_fun(Loss.GAMMA).mean(0) 
 
        J_z = (E_G_z - E_G_g)**2
        
        return -J_z.sum()

    @staticmethod
    def NegentropyLossNumpy(S, G_fun=None):
        if G_fun == None:
            G_fun = Loss.Logcosh
        return Loss.NegentropyLoss(torch.from_numpy(S), G_fun).numpy()

    @staticmethod
    def grad_norm(grad_old, grad_new):
        return torch.norm(grad_old - grad_new).cpu().detach().item()