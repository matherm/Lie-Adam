import torch
import torch.nn as nn
import math
from math import factorial as fac

def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom

class Bernstein():
    
    def __init__(self, n=3, a=torch.ones(4).unsqueeze(0)):
        self.n = n
        self.a=a
        self.basis = self.get_basis(self.n)
        self.derivative = self.get_derivatives()
         
    def get_basis(self, n):
        """
        Returns the n+1 Bernstein polynomials.
        """
        return [lambda x,i=i,n=n: binomial(n,i) * torch.pow(x, i) * torch.pow(1-x, n-i) for i in range(n+1)]

    def get_derivatives(self):
        """
        Returns the n+1 Bernstein polynomials.
        """
        n = self.n
        basis = self.get_basis(n-1)
        zero    = [lambda x,n=n: -n *  basis[0](x)]
        default = [lambda x,i=i:  n * (basis[i-1](x) - basis[i](x)) for i in range(1, n)]
        nth     = [lambda x,n=n:  n *  basis[n-1](x)]                               
        return zero + default + nth
        
    def __call__(self, x, a=None):
        """
        Returns the evaluated basis functions as Tensor with shape (B, m, values).
        
        Usage:
            y = bernstein(X).sum(1)
        """
        self.a = a if a is not None else self.a
        return torch.stack([a.unsqueeze(1) * Be_i(x) for a, Be_i in zip(self.a.T, self.basis)]).transpose(0,1)
    
    
    def df(self, x, a=None):
        """
        Returns the evaluated derivativess of the basis functions as Tensor with shape (B, m, values).
        
        Usage:
            y = bernstein(X).sum(1)
        """
        self.a = a if a is not None else self.a
        return torch.stack([a.unsqueeze(1) * Be_i(x) for a, Be_i in zip(self.a.T, self.derivative)]).transpose(1,0)
    
    def inverse(self, y, a=None, eps=1e-5, max_iters=1000):
        """
        Returns the inverse function values as Tensor with shape (B, m, values) by use of Newton iteration.
        
        x = f^(-1)(y)
        
        """
        self.a = a if a is not None else self.a
        n_curves, n_dims, n_samples = self.a.shape[0], y.shape[0], y.shape[1] 
        assert n_dims == n_curves
        
        # Function shortcuts for f, f' and error
        f  = lambda x: self(x).sum(1) - y
        df = lambda x: self.df(x).sum(1)
        J  = lambda x, x_ : torch.abs(x - x_) > eps
        
        # Initial state
        x_old = 0.5 + torch.zeros((n_curves, n_samples))
        x_new  = x_old - f(x_old) / df(x_old)
        not_converged = J(x_old, x_new)
               
        # Loop till convergence
        while not_converged.int().sum() > 0 and max_iters > 0:
            x_old[not_converged] = x_new[not_converged]
            x_new[not_converged]  = (x_old - f(x_old) / df(x_old))[not_converged]
            not_converged = J(x_old, x_new)
            max_iters -= 1
        return x_new


class BernsteinTransform(nn.Module): 
    """
    Transforms a 1D-Probability Distribution with a monotonic Bernstein Polynomial.
    """
    def __init__(self, n=3, n_neurons = 20, conditional=False):
        super().__init__()
        self.n = n
        self.bernstein = Bernstein(n)   
        self.sigm = torch.nn.Sigmoid()
        if conditional:
            #shared network
            self.nn = nn.Sequential(nn.Linear(1, n_neurons), nn.ReLU(), nn.Linear(n_neurons, n_neurons), nn.ReLU())  
            #scale and shift stage I
            self.nn_a = nn.Sequential(nn.Linear(n_neurons, 1), nn.Softplus())
            self.nn_b  = nn.Linear(n_neurons, 1) 
            #scale and shift stage II
            self.nn_alpha = nn.Sequential(nn.Linear(n_neurons, 1), nn.Softplus())
            self.nn_beta = nn.Linear(n_neurons, 1)       
            # bezier polynomials a.k.a. bernstein coefficients
            self.coeffs = nn.Sequential(nn.Linear(n_neurons, n_neurons), nn.ReLU(), nn.Linear(n_neurons, n+1))     
        else:
            # unconditional parameters
            self.coeffs = nn.Parameter(torch.distributions.Uniform(-1,1).sample((1,n+1)))
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))    
            
    @staticmethod
    def ascending(a):
        """
        Ensures monotony of the polynomial by increasing coefficients.
        """
        a1_n = nn.Softplus()(a[:, 1:])
        a = torch.cat([a[:,0:1], a1_n], dim=1)
        a = torch.cumsum(a, dim=1)
        return a
        
    def _condition(self, X=None): 
        """
        Conditions the parameters on X.
        """
        if not X is None:
            X = self.nn(X) 
            return [self.nn_a(X),
                      self.nn_b(X),
                      self.nn_alpha(X),
                      self.nn_beta(X),
                      BernsteinTransform.ascending(self.coeffs(X))]    
        else:
            return [self.a, self.b, self.alpha, self.beta, BernsteinTransform.ascending(self.coeffs)] 
          
    def forward(self, y, X):
        """
        Transforms the vector y
        """       
        a, b, alpha, beta, coeffs = self._condition(X)
        y_ = (y - b) / a                                                    # scale shift
        y_,  log_abs_dy_ = self.sigm(y_), -F.softplus(-y_) -F.softplus(y_)  # sigmoid, log-abs-sigmoid-derivative
        z = self.bernstein(y_, a=coeffs).sum(1)                             # Bernstein evaluate
        dz = self.bernstein.df(y_, a=coeffs).sum(1)                         # Bernstien derivative
        z_ = (z - beta) / alpha                                             # scale shift  
        
        self._log_abs_det = torch.log(1/torch.abs(a))  \
                            + torch.log(torch.abs(dz)) \
                            + torch.log(1/torch.abs(alpha)) \
                            + log_abs_dy_  
        return z_,  self._log_abs_det
     
    def inverse(self, z, X):
        """
        Inverse transform
        """
        a, b, alpha, beta, coeffs = self._condition(X)
        z_ = z * alpha + beta                                              # scale shift
        y_ = self.bernstein.inverse(z_, a=coeffs)                          # Bernstein
        y_ = y_.log() - (-y_).log1p()                                      # inv-sigmoid
        y =  y_ * a + b                                                    # scale shift
        return y

    def log_abs_det(self, x, y):
        return self._log_abs_det
    
import torch.nn.functional as F
class TransformationFlow(nn.Module): 
    
    def __init__(self, n=10, conditional=True):    
        super().__init__()
        
        # variables
        self.conditional = conditional   
        
        # bernstein bijector
        self.bernstein = BernsteinTransform(n=n,conditional=conditional)
            
    def forward(self, X): 
        return X

    def sample(self, X, size=100):
        """
        Samples y by transforming z ~ N(0,1).
        """
        z = torch.distributions.Normal(0, 1).sample((len(self.bernstein.params[0]), size))
        y = self.bernstein.inverse(z, X)
        return y.detach() 
        
    def log_prob(self, X, y):
        """
        Transforms y to z ~ N(0,1).
        """
        z_, log_abs_det = self.bernstein(y, X)
        log_prob = torch.distributions.Normal(0, 1).log_prob(z_)
        return log_prob + log_abs_det
    
    def fit(self, sample_data, lr=1e-3, epochs=300):    
        
        optim = torch.optim.Adam(self.parameters(), lr=lr)    
          
        for i in range(epochs*20):
            X, y, val = sample_data(20)
            X, y, val = torch.from_numpy(X),  torch.from_numpy(y),  torch.from_numpy(val)
            
            optim.zero_grad()
            
            if self.conditional:
                loss_ep = -self.log_prob(X,y).mean()
                loss_ep.backward()
                
                outpt = self(val[:,0:1])
                loss_val = -self.log_prob(X,val[:,1:2]).mean()
            else:
                outpt = self(None)
                loss_ep = -self.log_prob(None, y.reshape(1, -1)).mean()
                loss_ep.backward()
                
                loss_val = -self.log_prob(None, val.reshape(1, -1)).mean()
                        
            optim.step()
            if i % 400 == 0:
                print("iter:", i, "nll:", loss_ep.item(), "val:", loss_val.item())                   
                
                
class ParametricLoss(nn.Module):
    
    def __init__(self, n_components):
        super().__init__()
        self.flow_per_dim = nn.ModuleList([TransformationFlow(conditional=False) for i in range(n_components)])

    def __call__(self, s):
        """
        s (B, n_components) : the estimated components

        Return
            negative log probability (B, n_components)
        """
        return -torch.cat([self.flow_per_dim[i].log_prob(None, s[:,i:i+1]) for i in range(len(self.flow_per_dim))], dim=1)   