import torch
import numpy as np
from .expm32 import expm32
from .expm64 import expm64

def expm_frechet(A, E, expm):
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=A.dtype, device=A.device, requires_grad=False)
    M[:n, :n] = A
    M[n:, n:] = A
    M[:n, n:] = E
    return expm(M)[:n, n:]

class expm_class(torch.autograd.Function):
    @staticmethod
    def _expm_func(A):
        if A.element_size() > 4:
            return expm64
        else:
            return expm32

    @staticmethod
    def _expm_frechet(A, E):
        return expm_frechet(A, E, expm_class._expm_func(A))

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        expm = expm_class._expm_func(A)
        return expm(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return expm_class._expm_frechet(A.t(), G)

expm = expm_class.apply

def expm_caley(A):
    """
    Caley-Approximation for expm(A).
    """
    I = torch.eye_like(A)
    theta = A/2
    return torch.inv(I - theta) @ (I + theta)