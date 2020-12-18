import numpy as np    

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    """
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2colOrder(N, p):
    """
    Returns the indexes of an im2col block such that succesive items are patches of the same input image.
    
    Args:
        N (number) : the number of images
        p (number) : the number of patches
        ((np.arange(len(trn2)) * len(trn)) + np.repeat(np.arange(len(trn)), 4)) % len(trn2)
    """
    item_idx = np.arange(p) * N % p # the same input image repeats after N items i.e. [0, N, 2N, ...]
    patch_idx = np.repeat(np.arange(N), p//N) # index vector for the patches [0,0,..0,1,1,...1,1..,N,N]
    return  item_idx + patch_idx

def col2im(cols, x_shape,  BSZ=(5, 5), padding=0, stride=1, agg="sum"):
    """ An implementation of col2im based on fancy indexing and np.add.at with aggreation sum|avg"""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, BSZ[0], BSZ[1], padding,
                                stride)
    cols_reshaped = cols.reshape(C * BSZ[0] * BSZ[1], -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if agg == "avg":
        avg = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        np.add.at(avg, (slice(None), k, i, j), np.ones_like(cols_reshaped))
        x_padded /= avg
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def im2col(x,  BSZ=(5, 5), padding=0, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, BSZ[0], BSZ[1], padding,
                                stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(BSZ[0] * BSZ[1] * C, -1)
    return cols


def im2col2(X, BSZ=(5, 5), stepsize=1):
    """
    Converts a image like structure (H, W) to a structure with shape (patches, BSZ**2)

    Args:
        X : input tensor with image_like 

    Output:
        Tensor with shape (B x patches, C*components)
    """
    if type(BSZ) is not tuple:
        BSZ = (BSZ, BSZ)
    m,n = X.shape
    s0, s1 = X.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1
    out_view = np.lib.stride_tricks.as_strided(X, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


if __name__ == "__main__":

    from imageio import imread
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    X = imread("./data/dagm/Class1/Test/0001.PNG")[:500, :200]
    shape = X.shape
    print(shape)

    plt.imshow(X)
    plt.show()

    X = im2col(X.reshape(1,1,shape[0], shape[1]), BSZ=(5, 5), stride=1)
    print(X.shape)
   
    #model = PCA(whiten=True)
    #X = model.fit_transform(X.T)

    X = col2im(X, (1,1,shape[0], shape[1]), BSZ=(5, 5), stride=1)
    print(X.shape)
    
    plt.imshow(X[0,0])
    plt.show()