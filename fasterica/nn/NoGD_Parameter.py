import torch.nn as nn


class NoGDParameter(nn.Parameter):

    def __repr__(self):
        return f"{self.__class__.__name__}()"