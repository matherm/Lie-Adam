import torch
import torch.nn as nn
import math

from ..helpers import *
from ..nn import NoGDParameter, SOParameter

def isnan(x):
    return (x != x).sum() > 0

def skew_symmetric(x, triuidx, shape):
    skew = torch.zeros(*shape).to(x.device) 
    skew[triuidx[0], triuidx[1]] = x
    skew.T[triuidx[0], triuidx[1]] = -x
    return skew

class SGD_Lie(torch.optim.SGD):

    def step(self, closure=None):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
           
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, SOParameter):
                    triuidx = torch.triu_indices(*p.grad.shape, offset=1).to(p.grad.device)
                    d_p = p.grad.data[triuidx[0], triuidx[1]]
                    param_shape = p.data[triuidx[0], triuidx[1]]
                else:
                    d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if isinstance(p, SOParameter):
                    #  Compute the exponential_map of so(n)
                    direction = expm(-group['lr'] * skew_symmetric(d_p, triuidx, p.data.shape)).data  
                    p.data = direction.mm(p.data)
                elif isinstance(p, NoGDParameter):
                    p.data = p.grad.data.clone()
                else:
                    p.data.add_(-group['lr'], d_p)
