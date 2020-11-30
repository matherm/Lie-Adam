from hugeica import *
from sklearn.datasets import make_moons, load_digits

def nice_assert_st(A, B):
    if not A < B:
        raise AssertionError(f"Was {A}. Exepted {A} < {B}")

X = np.hstack([make_moons(n_samples=100)[0] for d in range(100)])
X = X - X.mean()
X = X / X.std()
X = X.astype(np.float32)

def test_whitening_cpu():
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    ica.cpu()

    ica.fit(dataloader, 1)
    nice_assert_st(Loss.FrobCov(X @ ica.unmixing_matrix) , 0.09)


def test_whitening_cuda():
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    ica.cuda()

    ica.fit(dataloader, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_numpy():
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    ica.fit(X, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_dataloader():
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    X_ = torch.from_numpy(X).float()
    dl = FastTensorDataLoader((X_,torch.empty(len(X))), batch_size=len(X))
    ica.fit(dl, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor():
    ica = FasterICA(n_components=2, optimistic_whitening_rate=1.0)    
    X_ = torch.from_numpy(X)
    ica.fit(X_, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor_optimistic():
    ica = FasterICA(n_components=2, optimistic_whitening_rate=0.9)    
    ica.fit(X, 1)
    assert Loss.FrobCov(X @ ica.unmixing_matrix) < 0.09

def test_whitening_tensor_gha():
    ica = FasterICA(n_components=2,  optimistic_whitening_rate=1.0, whitening_strategy="GHA")    
    ica.fit(X, 2)
    nice_assert_st(Loss.FrobCov(X @ ica.unmixing_matrix), 1)

def test_properties_gha():
    ica = FasterICA(n_components=2,  optimistic_whitening_rate=1.0, whitening_strategy="GHA")    
    ica.fit(X, 1)
    ica.unmixing_matrix
    ica.mixing_matrix
    ica.components_
    ica.sphering_matrix
    ica.explained_variance_

def test_properties_batch():
    ica = FasterICA(n_components=2,  optimistic_whitening_rate=1.0, whitening_strategy="batch")    
    ica.fit(X, 1)
    ica.unmixing_matrix
    ica.mixing_matrix
    ica.components_
    ica.sphering_matrix
    ica.explained_variance_

def test_bpd():
    ica = FasterICA(n_components=2,  optimistic_whitening_rate=1.0, whitening_strategy="batch")    
    ica.fit(X, 1)
    ica.bpd(X)

if __name__ == "__main__":

    test_whitening_cpu()
    test_whitening_cuda()
    test_whitening_numpy()
    test_whitening_dataloader()
    test_whitening_tensor()
    test_whitening_tensor_optimistic()
    test_whitening_tensor_gha()
    test_properties_gha()
    test_properties_batch()
    test_bpd()
    
