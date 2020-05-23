import torchvision
import numpy as np
import matplotlib.pyplot as plt
from fasterica import *

print("Fasterica version:", version)

cifar10 = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
X = np.vstack([cifar10[i][0].view(1, -1).numpy() for i in range(0, len(cifar10) )])

permut  = np.random.permutation(range(len(X)))
X_val = X[permut[:len(X) // 10]]
X     = X[permut[len(X) // 10:]]

X_val = X - X.mean()
X_val = X / X.std()
X = X - X.mean()
X = X / X.std()

ica = FasterICA(n_components=100, optimistic_whitening_rate=0.2)    

print("Fitting data shape", X.shape, "on", ica.device)
ica.fit(X, 10, X_val)

show_filters_color( ica.mixing_matrix.T)
plt.savefig("./cifar10-components.png")