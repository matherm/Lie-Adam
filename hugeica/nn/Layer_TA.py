import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter

from .Parameter import *
from .Loss import Loss

class TA_Layer(nn.Module):

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, X):
        B, F = X.shape
        return X.view(-1, self.T, F).mean(1)