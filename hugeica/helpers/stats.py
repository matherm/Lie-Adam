from scipy import stats 

def kurt(X, axis=0):
    return stats.kurtosis(X, axis=axis)

def skew(X, axis=0):
    return stats.skew(X, axis=axis)