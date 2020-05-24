import matplotlib.pyplot as plt
from .imagenet_loader import Imagenet

from fasterica import *

print("Fasterica version:", version)

imagenet = Imagenet(dataset="test", limit=100000)
dl = torch.utils.data.DataLoader(imagenet, num_workers=2, batch_size=20)

imagenetv = Imagenet(dataset="val", limit=500)
dlv = torch.utils.data.DataLoader(imagenetv, num_workers=2, batch_size=20)


model = FasterICA(25, loss="logcosh", optimistic_whitening_rate=0.2, whitening_strategy="GHA")

print("Fitting data shape", f"{len(imagenet)}x{imagenet[0][0].shape[1]}", "on", ica.device)
model.fit(dl, 30 , X_val=dlv)

show_filters_color(model.mixing_matrix.T)

plt.savefig("./imagenet-components.png")