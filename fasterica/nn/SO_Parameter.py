import torch.nn as nn


class SOParameter(nn.Parameter):

    def __repr__(self):
        return f"{self.__class__.__name__}()"