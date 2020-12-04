import numpy as np

def batch_regression(X, axis=1):
    """
    X \el (n, times, components)
    """
    def regr(x, y):
        """
        row-wise regression
        """
        n = x.shape[1]
        sum_xy = (x*y).sum(1)
        sum_y = y.sum(1)
        sum_x = x.sum(1)
        sum_x_sq = (x**2).sum(1)
        sq_sum_x = sum_x**2
        inter = ((sum_y * sum_x_sq) - (sum_x*sum_xy)) / (n*sum_x_sq - sq_sum_x)
        slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x_sq - sq_sum_x)
        assert len(slope) == len(x)
        return slope, inter
    
    n, t, c = X.shape
    X = X.transpose(0,2,1)
    # the function values
    X = X.reshape(n*c, t) 
    # the timesteps
    T = np.tile(np.linspace(-1, 1, t),(n*c, 1) ) 
    
    slope, inter = regr(T, X)
    slope = slope.reshape(n, c)
    inter = inter.reshape(n, c)  
    return slope, inter