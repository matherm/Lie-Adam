import torch.nn as nn

class Reshape(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, X):
        X = X.contiguous()
        return X.reshape(self.shape)