from sklearn.datasets import make_moons, load_digits
from fasterica import *

print("Fasterica version", version)

X = np.hstack([make_moons(n_samples=2000)[0] for d in range(100)])
X_val = np.hstack([make_moons(n_samples=500)[0] for d in range(100)])
X_val = X - X.mean()
X_val = X / X.std()
X = X - X.mean()
X = X / X.std()

dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float())
dataloader_valid = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

ica = FasterICA(n_components=10)    

print("Fitting data: ", X.shape, "on", ica.device)
ica.fit(dataloader, 10, dataloader_valid)