import torchvision
import numpy as np
import matplotlib.pyplot as plt
from fasterica import *

print("Fasterica version:", version)

stl = torchvision.datasets.STL10("./data", split='train+unlabeled', folds=None, transform=torchvision.transforms.ToTensor(), target_transform=None, download=True)
X = np.vstack([stl[i][0].view(1, -1).numpy() for i in range(0, len(stl) )])

permut  = np.random.permutation(range(len(X)))
X_val = X[permut[:len(X) // 10]]
X     = X[permut[len(X) // 10:]]

X_val = X - X.mean()
X_val = X / X.std()
X = X - X.mean()
X = X / X.std()

tensors = torch.from_numpy(X).float(), torch.empty(len(X))
dataloadert  = FastTensorDataLoader(tensors, batch_size=50)
tensors = torch.from_numpy(X_val).float(), torch.empty(len(X_val))
dataloaderv  = FastTensorDataLoader(tensors, batch_size=50)

ica = FasterICA(n_components=256, whitening_strategy="GHA")

print("Fitting data shape", X.shape, "on", ica.device)
ica.fit(dataloadert, 10, dataloaderv)

show_filters_color( ica.mixing_matrix.T)
plt.savefig("./stl-components.png")