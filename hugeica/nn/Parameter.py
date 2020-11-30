import torch.nn as nn


class NoGDParameter(nn.Parameter):

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class SOParameter(nn.Parameter):

    def __repr__(self):
        return f"{self.__class__.__name__}()"