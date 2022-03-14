import torch, warnings, scipy
import numpy as np
from .Loss import *
    
class AR1():
    def __init__(self, S, with_mean=True, with_std=True):
        self.n_timesteps = S.shape[1]
        self.n_components = S.shape[2]

        if not torch.is_tensor(S):
            S = torch.from_numpy(S)
        
        S = S.reshape(-1, self.n_components)
        S_change_score = (S.reshape(-1,  self.n_timesteps, self.n_components) + np.roll(S.reshape(-1,  self.n_timesteps, self.n_components), axis=1, shift=1))
        S_change_score = S_change_score.reshape(len(S), -1)

        self.std = S_change_score.std(0) if with_std else torch.ones(self.n_components)
        self.mean = S_change_score.mean(0) if with_mean else torch.zeros(self.n_components)

    def __call__(self, x):
        """
        log p(x_1,...,x_N) = log p(x_1) + \sum log p(x_n|x_n-1)
        """
        x = x.reshape(-1, self.n_timesteps, self.n_components)
        logprobs = Loss.ExpNormalized(x[:,0,:]).sum(1)
        for t in range(1, self.n_timesteps):
            change = x[:,t, :] - x[:,t-1, :]
            t = change
            logprobs += torch.distributions.Normal(self.mean, self.std).log_prob(t).sum(1)
        return logprobs.reshape(-1, 1) # mimic spatial dimension for elbo


        