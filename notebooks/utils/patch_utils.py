from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from torch.nn import functional as F
from skimage.transform import resize


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def flatten(X_patches):
    return X_patches.reshape(len(X_patches), -1)

def pool(X_patches):
    return X_patches.mean((2,3))


def i2c(X, p, stride=1):
    X_p = im2col(X, BSZ=(p, p), padding=0, stride=stride).T
    X_p = X_p[im2colOrder(len(X), len(X_p))] 
    return X_p

def cluster_centers_tiles(X, tile_size = 7, stride = 7):
    max_h = X.shape[2] # B, C, H, W
    means = []
    
    i = 0
    while (i)*stride + tile_size <=  max_h:    
        j = 0
        while (j)*stride + tile_size <=  max_h:
            X_ = X[:, :, i*stride:(i)*stride + tile_size, j*stride:(j)*stride + tile_size]
            mean = X_.reshape(len(X_), -1).mean(0)
            means.append(mean)
            j += 1
        i += 1            
        
    return np.stack(means)

def batch_resize(X, size=(28, 28)):
    n, c, h, w = X.shape
    X_out = []
    for i in range(n):
        X_out.append(resize(X[i], (c, *size)))
    return np.asarray(X_out)

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def concat_features(X, eff, layers=[4, 6], bs=50, fmap_pool=False, debug=False):
    
    m = torch.nn.AvgPool2d(3, 1, 1) if fmap_pool else lambda x : x
        
    net = eff[:layers[0]]
    with torch.no_grad():
        X_ = torch.cat([ m( net(torch.from_numpy( X[i:i+bs] ).to(device) )).detach().cpu() for i in range(0, len(X), bs)]).cpu()

    n, c, h, w = X_.shape
    
    if debug:
        print(X_.shape) 
    
    for l in layers[1:]:
        net = eff[:l]
        
        with torch.no_grad():
            X__ = torch.cat([ m( net(torch.from_numpy( X[i:i+bs] ).to(device) )).detach().cpu() for i in range(0, len(X), bs)]).cpu()
        
        if debug:
            print(X__.shape)
    
        #X__ = batch_resize(X__, size=(h, w))
        #X_ = np.concatenate([X_, X__], axis=1)
        X_ = embedding_concat(X_, X__)

    return X_.numpy()    

def get_vgg_features(X, bs=50):
    
    all_features = []
    
    model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
        
    def hook_t(module, input, output):
        m = torch.nn.AvgPool2d(3, 1, 1)
        features.append(m(output.cpu().detach()))
    
    model.layer2[-1].register_forward_hook(hook_t)
    model.layer3[-1].register_forward_hook(hook_t)
    
    for i in range(0, len(X), bs):
        features = []
        model(torch.from_numpy( X[i:i+bs] ).to(device) )
        all_features.append( embedding_concat(features[0], features[1] ) ) 
        
    return torch.cat(all_features)

def fmap_to_patches(X):
    if torch.is_tensor(X) == False:
        return fmap_to_patches( torch.from_numpy(X) ).numpy()
    
    B, F, H, W = X.shape
    X = X.transpose(1,2) # B, H, F, W
    X = X.transpose(2,3).contiguous() # B, H, W, F
    return X.view(B*H*W, F)


def patch_core_score_2(X_patches, X_test_patches, T, b=9, reweight=True):    
    # We need a knn structure
    neigh = NearestNeighbors(n_neighbors=b).fit(X_patches)
    
    # Compute the number of datapoints
    n = len(X_test_patches) // T
    
    # Find the nearest patchs
    dists, m = neigh.kneighbors(X_test_patches, return_distance=True) # n_test x k
    
    scores = []
    
    dists = dists.reshape(n, T, b )
    
    for i in range(n):
        score_patches = dists[i]
        N_b = score_patches[ np.argmax(score_patches[:, 0]) ] # the patch with the maximum distance
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b)))) if reweight else 1.
        score = w*np.max(score_patches[:,0]) # Image-level score
        scores.append(score)
        
    return scores

def patch_core_score(X_patches, X_test_patches, T, b=9, reweight=True):    
    # We need a knn structure
    neigh = NearestNeighbors(n_neighbors=1).fit(X_patches)
    
    # Compute the number of datapoints
    n = len(X_test_patches) // T
    
    # Find the nearest patchs
    dists, m = neigh.kneighbors(X_test_patches, return_distance=True) # n_test x 1
    
    # Get the maximum patch of a single image
    idx = dists.reshape((n, T)).argmax(1)
    s_ = dists.reshape((n, T))[range(n), idx]
    s_idx = np.arange(len(X_test_patches)).reshape(n, T)[range(n), idx]
    
    # Find the nearest neighbor of the maximum patch
    m_dists, m_idx = neigh.kneighbors(X_test_patches[s_idx], n_neighbors=1) 
    
    # Find the distances of the nearest neighbors to its b-nearest neighrbos
    mdists, mm = neigh.kneighbors(X_patches[m_idx.flatten()], n_neighbors=b, return_distance=True) # n_test x b
    
    # re-weight
    w = np.exp(mdists).sum(1) if reweight else 1.
    
    s = (1 - (np.exp(s_)/w)) * s_
    return s