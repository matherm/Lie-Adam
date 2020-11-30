import torch
import torch.nn as nn
import math
from .Loss import Loss

class Adaptive(nn.Module):

    def __init__(self, n_components, negative=False, trainable=False, G=Loss.Logcosh):
        super().__init__()
        if trainable:
            self.K = nn.Parameter(torch.randn(n_components))
        else:
            self.K = torch.ones(n_components)
        if negative:
            self.G = lambda x: -G(x)
        else:
            self.G = G

        self.trainable = trainable

    def __call__(self, s):
        if not self.trainable:
            self.K = torch.sign(Loss.K(s))
        else:
            self.K.data = torch.tanh(self.K)
        self.K = self.K.to(s.device)
        return self.K * self.G(s).to(s.device)