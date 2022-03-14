import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def show_images_with_error_bars(X, scores, c=3, bar_width=3):
    if scores.min() < 0:
        raise ValueError("Scores must be positive")
    if scores.min() > 1:
        raise ValueError("Scores must be less than 1")

    viridis = cm.get_cmap('hot', 100)        
    scores = (scores * 100).astype(np.int)
        
    h = w = int(np.sqrt(X.shape[1]/c))
    max_num = int(np.sqrt(len(X)))
    
    X = X.reshape(len(X), c, h, w).transpose(0, 2, 3, 1)
    rgb = viridis(scores)[:, :3].reshape(len(X), 1, 1, c) * np.ones((len(X), h, bar_width, c))
    
    for i in range(max_num**2):
        plt.subplot(max_num, max_num, i+1)
        plt.imshow(np.clip(np.hstack([rgb[i] , X[i]]), 0, 1))
        plt.axis("off")

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


def make_score_circle(images, scores, size=512, patch_size=32, shape=(3, 32, 32), scale=None):
    assert images.min() >= 0 and images.max() <= 1
    
    SHEET = np.zeros((size, size, 3)).astype(np.uint8)
    
    center = int(size//2)
    
    if scale==None:
        scale = scores.min(), scores.std()
    
    scores = (scores - scale[0]) / scale[1]
    scores =  scores * center/5 # 3sigma
    
    def preprocess(img):
        img = img.reshape(shape).transpose(1,2,0)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((patch_size, patch_size))
        img = np.asarray(img)
        assert img.shape[2] == 3
        return img
    
    coords = []
    for i in range(len(images)):   
        norm, alpha = scores[i], np.random.uniform(0, 2*np.pi)
        
        y = np.sin(alpha) * norm # sin_alpha = y/norm (sinussatz)
        y, x = int(y), int(np.sqrt(norm**2 - y**2)) # norm^2 = x^2 + y^2 (pythagoras)
        x = x * -1 if alpha > np.pi/2 and alpha < 3*np.pi/2 else x
                
        y = y + center
        x = x + center
        
        y, x = np.clip(y, 0, size-patch_size), np.clip(x, 0, size-patch_size)
        coords.append((y,x))
        
    # transform into 4 squares
    for i,(y,x) in enumerate(coords):
        img = images[i]
        img = preprocess(img)        
        
        SHEET[y:y+patch_size, x:x+patch_size] = img
        
    return SHEET, scale