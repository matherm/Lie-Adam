from fasterica import *
from sklearn.datasets import make_moons, load_digits

def test_whitening_cpu():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    ica.cpu()

    print("Fitting data: ", X.shape, "on", ica.device)
    ica.fit(dataloader, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09


def test_whitening_cuda():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    ica.cuda()

    print("Fitting data: ", X.shape, "on", ica.device)
    ica.fit(dataloader, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_numpy():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    print("Fitting data: ", X.shape, "on", ica.device)

    ica.fit(X, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_dataloader():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    print("Fitting data: ", X.shape, "on", ica.device)

    X = torch.from_numpy(X).float()
    dl = FastTensorDataLoader((X,torch.empty(len(X))), batch_size=len(X))
    ica.fit(dl, 1)
    assert Loss.FrobCov(X.numpy() @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor():
    return True
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    print("Fitting data: ", X.shape, "on", ica.device)

    X = torch.from_numpy(X)
    ica.fit(X, 1)
    assert Loss.FrobCov(X.numpy() @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor_optimistic():
    X = np.hstack([make_moons(n_samples=1000)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    ica = FasterICA(n_components=2, optimistic_whitening_rate=0.75)    

    print("Fitting data: ", X.shape, "on", ica.device)
    ica.fit(X, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09


if __name__ == "__main__":

    test_whitening_cpu()
    test_whitening_cuda()
    test_whitening_tensor_optimistic()
