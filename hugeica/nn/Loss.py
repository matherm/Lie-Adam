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
        return -torch.exp(-a2/2*x**2)/a2

    @staticmethod
    def Tanh(x):
        return torch.tanh(x)


    @staticmethod
    def RightSkew(x, a=1):
        """
        http://azzalini.stat.unipd.it/SN/skew-prop-aism.pdf
        Skewish log propability density function
        """
        gauss = torch.distributions.Normal(0,1)
        return np.log(2) + gauss.log_prob(x)  + torch.log(gauss.cdf(a*x))
        #return np.log(2) + gauss.log_prob(x)  + gauss.log_prob(a*x)

    @staticmethod
    def Hyper(x):
        """is log-probability
        """  
        # hat
        sech = lambda x: 1/torch.cosh(x)
        p = lambda x: 0.5*sech(x*np.pi/2)
        return torch.log(p(x))

    @staticmethod
    def Laplace(x):
        """is log-probability
        """
        # hat
        return np.log(0.5) - torch.abs(x) 

    @staticmethod
    def TemporalHyper(x, n_channels=43):
        """is log-probability
        Natural Image Statistics (10.12)
        shape (B, Channels*Pixels)
        """
        raise NotImplementedError()
        x = x.view(len(x), n_channels, -1)
        p_spatial = 2/(np.pi*3)*torch.exp(-np.sqrt(3)*torch.sqrt((x**2).sum(2)))
        log_p_channel = torch.log(p_spatial).sum(1)
        return log_p_channel

    @staticmethod
    def LogcoshNormalized(x):
        """is log-probability
        """
        # hat
        return -2*torch.log(torch.cosh(np.pi/(2*np.sqrt(3))*x)) - 4*np.sqrt(3)/(np.pi) + np.sqrt(2)

    @staticmethod
    def ExpNormalized(x):
        """is log-probability
        """
        # hat
        return torch.log(torch.exp(-np.sqrt(2)*torch.abs(x))/np.sqrt(2))

    @staticmethod
    def Gaussian(x):
        """is log-probability
        """
        return torch.distributions.Normal(0, 1).log_prob(x.flatten()).view(x.shape)

    @staticmethod
    def FrobCov(S):
        return np.linalg.norm(np.cov(S.T) - np.eye(S.shape[1])) / S.shape[1]

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
        S = S - S.mean(0)
        S = S / S.std(0)
        Loss.GAMMA = Loss.GAMMA.to(S.device)     
        E_G_z = G_fun(S).mean(0) 
        E_G_g = G_fun(Loss.GAMMA.repeat((1, S.shape[1]))).mean(0) 
 
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

    @staticmethod
    def amari(A, B):
        R = np.abs(A @ B)
        amari = ((R/R.max(0, keepdims=True)).sum(0)-1).sum() + ((R/R.max(1, keepdims=True)).sum(1)-1).sum()
        return amari

    @staticmethod
    def mcc(S, S_):
        mcc = np.abs(((S.T/np.linalg.norm(S.T,axis=0)).T @ (S_/np.linalg.norm(S_,axis=0)))).max(1).mean()
        return mcc
