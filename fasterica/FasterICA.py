import torch, warnings, time, sys
import torch.nn as nn
import numpy as np
import scipy
from tqdm import tqdm
from fasterica import *

class Net(nn.Module):

    def __init__(self, n_input, n_components, whiten=True, whitening_strategy="batch", dataset_size=-1, derivative="lie", fun=Loss.Logcosh):
        super().__init__()
        
        if whitening_strategy == "batch":
            self.whiten = Batch_PCA_Layer(n_input, n_components, dataset_size)
        if whitening_strategy == "GHA":
            self.whiten = HebbianLayer(n_input, n_components, dataset_size)
        
        if derivative == "lie":
            self.ica    = SO_Layer(n_components)
        elif derivative == "relative":
            self.ica    = Relative_Gradient(n_components, fun)
        else:
            ValueError(f"derivative={derivative} not understood.")

        if whiten:
            self.layers = nn.Sequential(self.whiten, self.ica)
        else:
            self.layers = nn.Sequential(self.ica)

    def forward(self, X):
        return self.layers(X)

class FasterICA(nn.Module):

    """
    tbd.
    """
    def __init__(self, n_components, whiten=True, loss="exp", optimistic_whitening_rate=0.5, whitening_strategy="batch", derivative="lie"):
        super().__init__()

        if whitening_strategy not in ["GHA", "batch"]:
            raise ValueError(f"Whitening strategy {whitening_strategy} not understood.")

        if whitening_strategy == "GHA":
            optimistic_whitening_rate = 1.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_components = n_components
        self.optimistic_whitening_rate = optimistic_whitening_rate
        self.derivative = derivative
        self.whiten = whiten
        self.whitening_strategy = whitening_strategy
        self.net = None
        self.optim = None
        self.history = []

        if loss == "logcosh":
            self.loss = Loss.Logcosh
        elif loss == "exp":
            self.loss = Loss.Exp
        elif loss == "parametric":
            self.loss = ParametricLoss(n_components)
        elif loss == "id":
            self.loss = Loss.Identity
        elif callable(loss):
            self.loss = loss
        else:
            raise ValueError(f"loss={loss} not understood.")

    def reset(self, input_dim, dataset_size, lr=1e-3):
        self.net = Net(input_dim, self.n_components, self.whiten, self.whitening_strategy, int(dataset_size * self.optimistic_whitening_rate), self.derivative, self.loss)
        if isinstance(self.loss, nn.Module):
            self.optim = Adam_Lie([{'params': self.net.whiten.parameters()},
                                   {'params': self.net.ica.parameters()},
                                   {'params': self.loss.parameters()}], lr=lr)
        else:
            self.optim = Adam_Lie([{'params': self.net.whiten.parameters()},
                                   {'params': self.net.ica.parameters()}], lr=lr)
        self.to(self.device)

    def reduce_lr(self, by=1e-1):
        if self.optim is not None:
            for param in self.optim.param_groups:
                param["lr"] = param["lr"] * by

    def cuda(self):
        self.to("cuda")
        return self
    
    def cpu(self):
        self.to("cpu")
        return self

    def to(self, device):
        self.device = device
        if self.net is not None:
            self.net.to(device)
        if isinstance(self.loss, nn.Module):
            self.loss.to(device)
        return self
    
    @property
    def unmixing_matrix(self, numpy=True):
        W_white = self.net.whiten.sphering_matrix 
        W_rot = self.net.ica.components_.T 
        return (W_white @ W_rot).cpu().detach().numpy()
    
    @property
    def mixing_matrix(self):
        return np.linalg.pinv(self.unmixing_matrix)

    @property
    def components_(self):
        return self.net.ica.components_.detach().cpu().numpy()

    @property
    def sphering_matrix(self, numpy=True):
        W_white = self.net.whiten.sphering_matrix 
        return W_white.cpu().detach().numpy()

    @property
    def explained_variance_(self):
        return self.net.whiten.explained_variance_.cpu().detach().numpy()

    def grad(self):
        if self.net is None:
            return 0.
        grad = [w.grad.flatten().detach() for w in self.net.parameters() if w.grad is not None and w.requires_grad]
        return torch.cat(grad).flatten().clone()

    def transform(X):
        if not torch.is_tensor(X):
            return self.transform(torch.from_numpy(X)).cpu().numpy()
        return self.net(X).detach()

    def _prepare_input(self, dataloader, validation_loader, bs):

        if bs == "auto":
            bs = self.n_components

        if bs < self.n_components and self.whiten:
            raise ValueError(f"Batch size ({bs}) too small. Expected batch size > n_components={self.n_components}")
        
        if isinstance(dataloader, np.ndarray):
            tensors =  torch.from_numpy(dataloader).float() , torch.empty(len(dataloader))
        if torch.is_tensor(dataloader):
            tensors = dataloader.float(), torch.empty(len(dataloader))
        if not isinstance(dataloader, torch.utils.data.DataLoader):        
            dataloader  = FastTensorDataLoader(tensors, batch_size=bs)
        
        if validation_loader is None:
            validation_loader = dataloader
        else:
            if isinstance(validation_loader, np.ndarray):
                tensors =  torch.from_numpy(validation_loader).float() , torch.empty(len(validation_loader))
            if torch.is_tensor(validation_loader):
                tensors = validation_loader.float(), torch.empty(len(validation_loader))
            if not isinstance(validation_loader, torch.utils.data.DataLoader):   
                validation_loader = FastTensorDataLoader(tensors, batch_size=bs)

        return dataloader, validation_loader

    def fit(self, X, epochs=10, X_val=None, lr=1e-3, bs="auto"):

        dataloader, validation_loader = self._prepare_input(X, X_val, bs)
        dataset_size = len(dataloader) * dataloader.batch_size
        t_start = time.time()
          
        def fit(ep):
            if ep == 0:
                iterator = zip(dataloader, tqdm(range(len(dataloader)), file=sys.stdout))
            else:
                iterator = zip(dataloader, range(len(dataloader)))
            if ep == int(0.9 * epochs):
                self.reduce_lr()
            losses = 0.
            for batch, _ in iterator:
                data, label = batch[0].to(self.device), None
                
                if self.net is None: 
                    self.reset(data.shape[1], dataset_size, lr)
                
                self.optim.zero_grad()
                output = self.net(data)
                loss = self.loss(output).mean(1).mean()
                loss.backward()
                self.optim.step()
                losses += loss.detach()

            if isinstance(self.net.whiten, HebbianLayer):
                self.net.whiten.step(ep, lr, self.optim.param_groups[0])
            return  losses.cpu().item()/len(dataloader)
            
        def evaluate(ep, train_loss):
            loss = 0.
            datalist = []
            t0 = time.time()
            for batch in validation_loader:
                data, label = batch[0].to(self.device), None
                output = self.net(data)
                loss += self.loss(output).mean(1).mean().detach()
                datalist.append(output.detach())
            S = torch.cat(datalist, dim=0).cpu().numpy()
            loss = ep, loss.cpu().item()/len(validation_loader), Loss.FrobCov(S), Loss.Kurtosis(S), Loss.MI_negentropy(S), time.time() - t_start, Loss.grad_norm(grad_old, self.grad())
            print(f"Ep.{ep:3} - {train_loss:.2f} - validation (loss/white/kurt/mi): {loss[1]:.2f} / {loss[2]:.2f} / {loss[3]:.2f} / {loss[4]:.2f} (eval took: {time.time() - t0:.1f}s)")
            self.history.append(loss)
            
        for ep in range(epochs):
            grad_old = self.grad()
            train_loss = fit(ep)
            evaluate(ep, train_loss)

        return self.history