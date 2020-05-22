import torch, warnings
import torch.nn as nn
import numpy as np
import scipy
from fasterica import *

class Net(nn.Module):

    def __init__(self, n_input, n_components, whiten=True, dataset_size=-1):
        super().__init__()
        self.whiten = Batch_PCA_Layer(n_input, n_components, dataset_size)
        self.ica    = SO_Layer(n_components)
        if whiten:
            self.layers = nn.Sequential(self.whiten, self.ica)
        else:
            self.layers = nn.Sequential(self.ica)

    def forward(self, X):
        return self.layers(X)

class FasterICA():

    """
    tbd.
    """
    def __init__(self, n_components, whiten=True, loss="exp", optimistic_whitening_rate=1.0):

        self.device = "cpu"
        self.optimistic_whitening_rate = optimistic_whitening_rate
        self.n_components = n_components
        self.whiten = whiten
        self.net = None

        if loss == "logcosh":
            self.loss = Loss.Logcosh
        elif loss == "exp":
            self.loss = Loss.Exp
        else:
            raise ValueError(f"loss={loss} not understood.")

    def reset(self, input_dim, dataset_size, lr=1e-3):
        self.net = Net(input_dim, self.n_components, self.whiten, int(dataset_size * self.optimistic_whitening_rate))
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
        W_white = self.net.whiten.weight.T / torch.sqrt(self.net.whiten.bias)
        W_rot = self.net.ica.weight.T 
        return (W_white @ W_rot).cpu().detach().numpy()
    
    @property
    def mixing_matrix(self):
        return np.linalg.pinv(self.mixing_matrix)

    @property
    def components_(self):
        return self.net.ica.weight.T.detach().cpu().numpy()

    @property
    def sphering_matrix(self, numpy=True):
        W_white = self.net.whiten.weight.T / torch.sqrt(self.net.whiten.bias)
        return (W_white @ W_rot).cpu().detach().numpy()

    @property
    def explained_variance_(self):
        return self.net.whiten.bias.detach().cpu().numpy()

    def transform(X):
        if not torch.is_tensor(X):
            return self.transform(torch.from_numpy(X)).cpu().numpy()
        return self.net(X).detach()

    def fit(self, dataloader, epochs=10, validation_loader=None, lr=1e-3):
        
        if isinstance(dataloader, np.ndarray):
            dataloader = FastTensorDataLoader((torch.from_numpy(dataloader).float(),torch.empty(len(dataloader))))
        if torch.is_tensor(dataloader):
            dataloader = FastTensorDataLoader((dataloader.float(), torch.empty(len(dataloader))))

        if validation_loader is None:
            validation_loader = dataloader

        dataset_size = len(dataloader) * dataloader.batch_size
        
        def fit(ep):
            for batch in dataloader:
                data, label = batch[0].to(self.device), None
                
                if self.net is None: 
                    self.reset(data.shape[1], dataset_size, lr)
                
                self.optim.zero_grad()
                output = self.net(data)
                loss = self.loss(output).sum(1).mean()
                loss.backward()
                self.optim.step()

        def evaluate(ep):
            loss = 0.
            datalist = []
            for batch in validation_loader:
                data, label = batch[0].to(self.device), None
                output = self.net(data)
                loss += self.loss(output).sum(1).mean().detach()
                datalist.append(data.detach())
            S = torch.cat(datalist, dim=0).cpu().numpy() @  self.unmixing_matrix
            print(f"Eval ep.{ep:3} - validation (loss/white/kurt/mi): {loss/len(validation_loader):.2f} / {Loss.FrobCov(S):.2f} / {Loss.Kurtosis(S):.2f} / {Loss.MI(S):.2f}")

        for ep in range(epochs):
            fit(ep)
            evaluate(ep)

        return self