from fasterica import *
from sklearn.datasets import make_moons, load_digits

def test_whitening_cpu():
    X = np.hstack([make_moons(n_samples=2000)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    ica = FasterICA(n_components=2)    

    device = "cpu"
    if torch.cuda.is_available():
        ica.cuda()
        device = "cuda"

    print("Fitting data: ", X.shape, "on", device)
    ica.fit(dataloader, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09


def test_whitening_cuda():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    ica = FasterICA(n_components=2)    

    device = "cpu"
    if torch.cuda.is_available():
        ica.cuda()
        device = "cuda"


    print("Fitting data: ", X.shape, "on", device)
    ica.fit(dataloader, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    ica = FasterICA(n_components=2)    

    device = "cpu"
    if torch.cuda.is_available():
        ica.cuda()
        device = "cuda"

    print("Fitting data: ", X.shape, "on", device)
    ica.fit(X, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor_optimistic():
    X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
    X = X - X.mean()
    X = X / X.std()

    ica = FasterICA(n_components=2, optimistic_whitening_rate=0.5)    

    device = "cpu"
    if torch.cuda.is_available():
        ica.cuda()
        device = "cuda"

    print("Fitting data: ", X.shape, "on", device)
    ica.fit(X, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09


if __name__ == "__main__":

    test_whitening_cpu()
    test_whitening_cuda()
