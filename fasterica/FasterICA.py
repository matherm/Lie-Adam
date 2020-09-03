import torch, warnings, time, sys
import torch.nn as nn
import numpy as np
import itertools
import scipy
from tqdm import tqdm
from fasterica import *

class Net(nn.Module):

    def __init__(self, n_input, n_components, whiten=True, whitening_strategy="batch", dataset_size=-1, derivative="lie", fun=Loss.Logcosh):
        super().__init__()
        
        if whiten:
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
    def __init__(self, n_components, whiten=True, loss="exp", optimistic_whitening_rate=0.5, whitening_strategy="batch", derivative="lie", optimizer="adam", reduce_lr=False):
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
        self.optimitzer = optimizer
        self.optim = None
        self.history = []
        self._reduce_lr = reduce_lr
        self.K = torch.ones(n_components)
        if loss == "logcosh":
            self.G = Loss.Logcosh
            self.loss = lambda x: -self.G(x)      
        elif loss == "neglogcosh":
            self.G = lambda x : -Loss.Logcosh(x)
            self.loss = lambda x: -self.G(x)      
        elif loss == "exp":
            self.G = Loss.Exp
            self.loss = lambda x: -self.G(x)      
        elif loss == "negexp":
            self.G = lambda x : -Loss.Exp(x)
            self.loss = lambda x: -self.G(x)      
        elif loss == "tanh":
            self.G = Loss.Tanh
            self.loss = lambda x: -self.G(x)      
        elif loss == "parametric":
            self.G = ParametricLoss(n_components) 
            self.loss = lambda x: -self.G(x)      
        elif loss == "nnica":
            self.G = Loss.NNICA
            self.loss = lambda x : Loss.NegentropyLoss(x, self.G)
        elif loss == "id":
            self.G = Loss.Identity
            self.loss = self.G
        elif loss == "adaptivelogcoshK":
            self.G = Adaptive(n_components, G=Loss.Logcosh)
            self.loss = self.G
        elif loss == "adaptiveexpK":
            self.G = Adaptive(n_components, G=Loss.Exp)
            self.loss = self.G
        elif loss == "negentropy_logcosh":
            self.G = Loss.Logcosh
            self.loss = lambda x : Loss.NegentropyLoss(x, self.G)
        elif loss == "negentropy_exp":
            self.G = Loss.Exp
            self.loss = lambda x : Loss.NegentropyLoss(x, self.G)
        elif loss == "negentropy_tanh":
            self.G = Loss.Tanh
            self.loss = lambda x : Loss.NegentropyLoss(x, self.G)
        elif loss == "ISA":
            self.G = ISA(n_components, 16, layout="topogrid")
            self.loss = lambda x : self.G(x)
        elif callable(loss):
            self.G = loss
            self.loss = lambda x : -self.G(x)   
        else:
            raise ValueError(f"loss={loss} not understood.")

    def reset(self, input_dim, dataset_size, lr=1e-3):
        self.net = Net(input_dim, self.n_components, self.whiten, self.whitening_strategy, int(dataset_size * self.optimistic_whitening_rate), self.derivative, self.G)
        if self.optimitzer == "adam":
            if isinstance(self.G, nn.Module):
                if self.whiten:
                    self.optim = Adam_Lie([{'params': self.net.whiten.parameters(), "lr" : lr},
                                           {'params': self.net.ica.parameters(), "lr" : lr},
                                           {'params': self.G.parameters(), "lr" : lr}], amsgrad=False)
                else:
                    self.optim = Adam_Lie([{'params': self.net.ica.parameters(), "lr" : lr},
                                           {'params': self.G.parameters(), "lr" : lr}], amsgrad=False)
            else:
                if self.whiten:
                    self.optim = Adam_Lie([{'params': self.net.whiten.parameters(), "lr" : lr},
                                           {'params': self.net.ica.parameters(), "lr" : lr}], amsgrad=False)
                else:
                    self.optim = Adam_Lie([{'params': self.net.ica.parameters(), "lr" : lr}], amsgrad=False)
        elif self.optimitzer == "sgd":
            if isinstance(self.G, nn.Module):
                if self.whiten:
                    self.optim = SGD_Lie([{'params': self.net.whiten.parameters(), "lr" : lr},
                                           {'params': self.net.ica.parameters(), "lr" : lr},
                                           {'params': self.G.parameters(), "lr" : lr}])
                else:
                    self.optim = SGD_Lie([{'params': self.net.ica.parameters(), "lr" : lr},
                                           {'params': self.G.parameters(), "lr" : lr}])
            else:
                if self.whiten:
                    self.optim = SGD_Lie([{'params': self.net.whiten.parameters(), "lr" : lr},
                                           {'params': self.net.ica.parameters(), "lr" : lr}])
                else:
                    self.optim = SGD_Lie([{'params': self.net.ica.parameters(), "lr" : lr}])
        else:
            gens = itertools.chain()
            gens = itertools.chain(gens,  self.net.ica.parameters())
            if self.whiten:
                gens = itertools.chain(gens,  self.net.whiten.parameters())
            if isinstance(self.G, nn.Module):
                gens = itertools.chain(gens,  self.loss.parameters())
            self.optim = LBFGS_Lie(gens, lr=lr, line_search_fn="strong_wolfe")
        self.to(self.device)

    def reduce_lr(self, by=1e-1):
        if self.optim is not None and self._reduce_lr:
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
        if isinstance(self.G, nn.Module):
            self.G.to(device)
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
        if len(grad) == 0:
            return 0.
        return torch.cat(grad).flatten().clone()

    def transform(self, X):
        if not torch.is_tensor(X):
            return self.transform(torch.from_numpy(X)).cpu().numpy()
        device = next(self.parameters()).device
        return self.net(X.to(device)).detach()

    def forward(self, X):
        if not torch.is_tensor(X):
            return self.forward(torch.from_numpy(X)).cpu().numpy()
        device = next(self.parameters()).device
        return self.net(X.to(device))

    def score_norm(self, X, ord=0.5):
        if not torch.is_tensor(X):
            return self.score_norm(torch.from_numpy(X)).cpu().numpy()
        device = next(self.parameters()).device
        s = self.transform(X.to(device))
        return np.norm(s, axis=1, ord=ord)
    
    def score(self, X, ord=0.5):
        if not torch.is_tensor(X):
            return self.score(torch.FloatTensor(X)).cpu().numpy()
        device = next(self.parameters()).device
        s = self.net(X).to(device)
        return self.loss(s).mean(1)
    
    def _prepare_input(self, dataloader, validation_loader, bs):

        if bs == "auto":
            bs = self.n_components

        if bs < self.n_components and self.whiten and self.whitening_strategy == "batch":
            raise ValueError(f"Batch size ({bs}) too small. Expected batch size > n_components={self.n_components}")
        
        if isinstance(dataloader, np.ndarray):
            tensors =  torch.from_numpy(dataloader) , torch.empty(len(dataloader))
        if torch.is_tensor(dataloader):
            tensors = dataloader, torch.empty(len(dataloader))
        if not isinstance(dataloader, torch.utils.data.DataLoader):        
            dataloader  = FastTensorDataLoader(tensors, batch_size=bs)
        
        if validation_loader is None:
            validation_loader = dataloader
        else:
            if isinstance(validation_loader, np.ndarray):
                tensors =  torch.from_numpy(validation_loader) , torch.empty(len(validation_loader))
            if torch.is_tensor(validation_loader):
                tensors = validation_loader, torch.empty(len(validation_loader))
            if not isinstance(validation_loader, torch.utils.data.DataLoader):   
                validation_loader = FastTensorDataLoader(tensors, batch_size=bs)

        return dataloader, validation_loader

    def fit(self, X, epochs=10, X_val=None, lr=1e-3, bs="auto", logging=-1):

        dataloader, validation_loader = self._prepare_input(X, X_val, bs)
        dataset_size = len(dataloader) * dataloader.batch_size
        t_start = time.time()
          
        def fit(ep):
            if ep == 0 and logging > 0:
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
                loss = self.loss(output).mean()
                loss.backward()
                losses += loss.detach()
                self.optim.step(lambda : self.loss(self.net(data.detach())).mean())
            if hasattr(self.net, "whiten") and isinstance(self.net.whiten, HebbianLayer):
                self.net.whiten.step(ep, lr, self.optim.param_groups[0])
            return  losses.cpu().item()/len(dataloader)  
            
        def evaluate(ep, train_loss):
            loss = 0.
            datalist = []
            t0 = time.time()
            for batch in validation_loader:  
                data, label = batch[0].to(self.device), None
                output = self.net(data).detach()
                loss += self.loss(output).mean()
                datalist.append(output)
            S, S_ = torch.cat(datalist, dim=0), torch.cat(datalist, dim=0).cpu().numpy()
            loss =(ep,                                       # 0
                   loss.cpu().item()/len(validation_loader), # 1
                   Loss.FrobCov(S_),                         # 2
                   Loss.Kurtosis(S_),                        # 3
                   Loss.NegentropyLoss(S, self.G),           # 4    
                   time.time() - t_start,                    # 5
                   Loss.grad_norm(grad_old, self.grad()),    # 6
                   Loss.Logprob(S, self.G) if not isinstance(self.loss, nn.Module) else 0., # 7
                   train_loss)                               # 8
            print(f"Ep.{ep:3} - {train_loss:.4f} - validation (loss/white/kurt/mi/logp): {loss[1]:.4f} / {loss[2]:.2f} / {loss[3].mean():.2f} / {loss[4]:.4f} / {loss[7]:.4f} (eval took: {time.time() - t0:.1f}s)")
            self.history.append(loss)
            
        for ep in range(epochs):
            grad_old = self.grad()
            train_loss = fit(ep)
            if ep+1 % logging == 0 and logging > 0 or logging == 1:
                evaluate(ep, train_loss)

        return self.history
