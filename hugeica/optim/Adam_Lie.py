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

class Adam_Lie(torch.optim.Adam):

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, SOParameter):
                    triuidx = torch.triu_indices(*p.grad.shape, offset=1).to(p.grad.device)
                    grad = p.grad.data[triuidx[0], triuidx[1]]
                    param_shape = p.data[triuidx[0], triuidx[1]]
                else:
                    grad = p.grad.data
                    param_shape = p.data  

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param_shape)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param_shape)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(param_shape)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if isinstance(p, SOParameter):
                    #  Compute the exponential_map of so(n)
                    direction = expm( skew_symmetric(-step_size/denom * exp_avg, triuidx, p.data.shape)).data  
                    p.data = direction.mm(p.data)
                elif isinstance(p, NoGDParameter):
                    p.data = p.grad.data.clone()
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss