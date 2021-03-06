import torch
import numpy as np

def scatter_mean(batch, vals, unique=None):
    """
    #scatter_mean(np.array([0, 0, 0, 0, 0, 2, 2, 2]), np.array([[2, 2, 2, 2, 4, 3, 3, 3],[2, 2, 2, 2, 4, 3, 3, 3]]).T, unique=4)
    batch = [0, 0, 0, 0, 0, 1, 1, 1]
    vals =  [2, 2, 2, 2, 3, 3, 3, 4]
    vals2 = [1, 1, 1, 1, 1, 1, 1, 1]

    target = np.zeros(2)
    np.add.at(target, batch, vals)

    target2 = np.zeros(2)
    np.add.at(target2, batch, vals2)

    target/target2 = array([2.2       , 3.33333333])
    """
    if unique is None:
        n = len(np.unique(batch))
    else:
        n = unique
    d = vals.shape[1]
    
    sums = np.zeros((n, d))
    np.add.at(sums, batch, vals)

    m = np.zeros((n))
    np.add.at(m, batch, np.ones(len(batch)))
    m = m.reshape(-1, 1)
    
    avg = np.divide(sums, m, out=np.zeros_like(sums), where=m!=0)

    return avg


def sum_of_diff(X, axis=1):
    if torch.is_tensor(X):
        return torch.FloatTensor(sum_of_diff(X.cpu().detach().numpy(), axis)).to(X.device)
    return np.diff(X, axis=axis).sum(axis)

def incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """ 
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
    variance: recommendations, The American Statistician, Vol. 37, No. 3,
    pp. 242-247
    """
    last_sum = last_mean * last_sample_count
    new_sum = X.sum(0, keepdim=True)
    new_sample_count = len(X)
    updated_sample_count = last_sample_count + new_sample_count
    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = X.var(0, keepdim=True) * new_sample_count
        last_unnormalized_variance = last_variance * last_sample_count
    
        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
                last_unnormalized_variance + new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count
    return updated_mean, updated_variance, updated_sample_count


def svd_flip(u, v):
    """
    https://github.com/scikit-learn/scikit-learn/blob/483cd3eaa/sklearn/decomposition/_incremental_pca.py
    """
    max_abs_rows = torch.argmax(torch.abs(v), axis=1)
    signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
    n_some = u.shape[1]
    u = u * signs[:n_some]
    v = v * signs.unsqueeze(1)
    return u, v