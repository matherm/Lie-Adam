import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def from_numpy(x):
    return torch.from_numpy(x)

def toPIL(x):
    if not torch.is_tensor(x):
        return toPIL(torch.from_numpy(x))
    return TF.to_pil_image(x)

def toTensor(x):
    return TF.to_tensor(x)

def toImage(x):
    if torch.is_tensor(x):
        return x.transpose(0,1).transpose(1,2)
    return x.transpose(1,2,0)

def toNumpy(x):
    return np.asarray(x)

def centerCrop(x, size=64):
    return TF.center_crop(x, size)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = np.asarray(img).transpose(1,2,0)
        img = img - img.min()
        img = img / img.max()
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])