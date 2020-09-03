import matplotlib.pyplot as plt
import numpy as np

def show_filters_color(W, space=1, C=3):
    C, M, N = C, W.shape[0]/C, W.shape[1]
    shape = (C, int(np.sqrt(M)), int(np.sqrt(M)))
    
    grid = int(np.sqrt(N))
    im_c, im_h, im_w = C, (shape[1] + space), (shape[2] + space)
    
    IM = np.zeros( (C, grid * im_h, grid * (im_w)) )

    for i in range(grid):
        for j in range(grid):
            im_padded = np.ones((im_c, im_h, im_w))
            w = W.T[i+(j*grid)].reshape(shape)
            w = (w - w.min()) / (w.max() - w.min())
            im_padded[:, :shape[1], :shape[2]] = w
            IM[:, i*im_h: (i+1)*im_h, j*im_w: (j+1)*im_w] = im_padded
    IM = IM.transpose(1,2,0).squeeze()
    if IM.ndim == 3:
        plt.imshow(IM)
    else:
        plt.imshow(IM, cmap="gray")
        
    plt.axis('off')