import torch.nn as nn

class Transpose(nn.Module):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, X):
        return X.transpose(self.a, self.b)