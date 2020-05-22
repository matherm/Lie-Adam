import torch, warnings
import torch.nn as nn
import numpy as np
import scipy
from fasterica import *

class Net(nn.Module):

    def __init__(self, n_input, n_components, whiten=True, dataset_size=-1):
        super().__init__()
        
        self.layer_whitening = Batch_PCA_Layer(n_input, n_components, dataset_size)
        self.layer_ica = SO_Layer(n_components)
        self.whiten = whiten

    def forward(self, X):
        if self.whiten:
            X = self.layer_whitening(X)
        X = self.layer_ica(X)
        return X

class Loss():

    @staticmethod
    def logcosh(x, a1=1.2):
        """is proportional negative log-likelihood
        """
        # inverse soft hat
        ax = a1*x
        if ax.max() > 80 or ax.min() < -80:
            warnings.warn(f"Exceeding range of cosh(). Maybe decrease a1: ({a1})")
            ax = torch.clamp(ax, -80, 80)
        return -(-torch.log( torch.cosh( ax ) + 1e-5 )/a1)

    @staticmethod
    def exp(x, a2=0.99):
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
    

class FasterICA():

    """
    tbd.
    """
    def __init__(self, n_components, whiten=True, loss="exp"):

        self.device = "cpu"
        self.n_components = n_components
        self.whiten = whiten
        self.net = None

        if loss == "logcosh":
            self.loss = Loss.logcosh
        elif loss == "exp":
            self.loss = Loss.exp
        else:
            raise ValueError(f"loss={loss} not understood.")

    def init(self, batch, dataset_size, lr=1e-3):
        input_dim = batch.shape[1]
        self.net = Net(input_dim, self.n_components, self.whiten, dataset_size)
        self.optim = Adam_Lie(self.net.parameters(), lr=lr)
        self.net.to(self.device)

    def cuda(self):
        self.to("cuda")
    
    def cpu(self):
        self.to("cpu")

    def to(self, device):
        self.device = device
    
    @property
    def unmixing_matrix(self, numpy=True):
        W_white = self.net.layer_whitening.weight.T / torch.sqrt(self.net.layer_whitening.bias)
        W_rot = self.net.layer_ica.weight.T 
        if numpy:
            return (W_white @ W_rot).cpu().detach().numpy()
        return (W_white @ W_rot).cpu().detach()
    
    @property
    def mixing_matrix(self):
        return np.linalg.pinv(self.mixing_matrix)

    @property
    def components_(self):
        W_white = self.net.layer_whitening.weight.T / torch.sqrt(self.net.layer_whitening.bias)
        return self.net.layer_ica.weight.T.detach().cpu().numpy()

    @property
    def explained_variance_(self):
        return self.net.layer_whitening.bias.detach().cpu().numpy()


    def transform(X):
        if not torch.is_tensor(X):
            return self.transform(torch.from_numpy(X)).cpu().numpy()
        return self.net(X).detach()

    def fit(self, dataloader, epochs=10, validation_loader=None, lr=1e-3):

        dataset_size = len(dataloader) * dataloader.batch_size

        if validation_loader is None:
            validation_loader = dataloader

        def fit_epoch():
            for batch in dataloader:
                data, label = batch[0].to(self.device), None
                
                if self.net is None:
                    self.init(data, dataset_size, lr)

                self.optim.zero_grad()
                output = self.net(data)
                loss = self.loss(output).sum(1).mean()
                loss.backward()
                self.optim.step()

        def evaluate():
            loss = 0.
            datalist = []
            for batch in validation_loader:
                data, label = batch[0].to(self.device), None
                output = self.net(data)
                loss += self.loss(output).sum(1).mean().detach()
                datalist.append(data.detach())
            datalist = torch.cat(datalist, dim=0).cpu()
            print(f"Eval: Validation loss (ica/white/kurt): {loss/len(validation_loader):.2f} / {Loss.FrobCov(datalist.numpy(), self.unmixing_matrix):.2f} / {Loss.Kurtosis(datalist.numpy(), self.unmixing_matrix):.2f}")

        for ep in range(epochs):
            fit_epoch()
            evaluate()

        return self