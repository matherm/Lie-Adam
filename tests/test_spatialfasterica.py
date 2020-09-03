from fasterica import *
from sklearn.datasets import make_moons, load_digits

def nice_assert_st(A, B):
    if not A < B:
        raise AssertionError(f"Was {A}. Exepted {A} < {B}")

X = np.hstack([make_moons(n_samples=3072)[0] for d in range(3072//2)])
X = X - X.mean()
X = X / X.std()
X = X.astype(np.float32)

X_valid = np.hstack([make_moons(n_samples=3072)[0] for d in range(3072//2)])
X_valid = X_valid - X_valid.mean()
X_valid = X_valid / X_valid.std()
X_valid = X_valid.astype(np.float32)

def test_fit():
    for dev in ["cuda", "cpu"]:
        ica = SpatialICA(shape=(3,32,32), 
                        BSZ=(16, 16), 
                        padding=0, 
                        stride=4, 
                        n_components=16, 
                        loss="negexp", 
                        optimistic_whitening_rate=1000, 
                        whitening_strategy="batch", 
                        reduce_lr=True)
        ica.to(dev)
        ica.fit(X, 1, X_val=X_valid, logging=-1, lr=1e-2, bs=10000)
        ica.unmixing_matrix
        ica.mixing_matrix
        ica.components_
        ica.sphering_matrix
        ica.explained_variance_

def test_score():
    ica = SpatialICA(shape=(3,32,32), 
                    BSZ=(16, 16), 
                    padding=0, 
                    stride=4, 
                    n_components=16, 
                    loss="negexp", 
                    optimistic_whitening_rate=1000, 
                    whitening_strategy="batch", 
                    reduce_lr=True)
    ica.fit(X, 1, X_val=X_valid, logging=-1, lr=1e-2, bs=10000)
    s = ica.score(X)
    assert s.shape == (len(X),)
    s = ica.score(torch.FloatTensor(X))
    assert s.shape == (len(X),)


def test_score_norm():
    ica = SpatialICA(shape=(3,32,32), 
                    BSZ=(16, 16), 
                    padding=0, 
                    stride=4, 
                    n_components=16, 
                    loss="negexp", 
                    optimistic_whitening_rate=1000, 
                    whitening_strategy="batch", 
                    reduce_lr=True)
    ica.fit(X, 1, X_val=X_valid, logging=-1, lr=1e-2, bs=10000)
    s = ica.score_norm(X)
    assert s.shape == (len(X),)
    s = ica.score_norm(torch.FloatTensor(X))
    assert s.shape == (len(X),)

if __name__ == "__main__":

    test_score()
    test_score_norm()
    test_fit()
