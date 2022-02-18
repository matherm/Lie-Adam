from sklearn.neighbors import NearestNeighbors
import torch
import time
import numpy as np
from torch.nn import functional as F
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from hugeica.helpers.kcenter_greedy import *


def tiles_to_fmap(S_fmap):
    """
    in (n, n_tiles, n_components)
    
    returns (n, n_components, h, w)
    """
    n_tiles = S_fmap.shape[1]
    hw = int(np.sqrt(n_tiles))
    n = len(S_fmap)
    c = S_fmap.shape[2]

    return S_fmap.reshape(n, n_tiles, c).transpose(0, 2, 1).copy().reshape(n, c, hw, hw)  # reshape to feature map

def avg_pool(fmap, size=1):
    return F.avg_pool2d(torch.from_numpy(fmap), size, stride=1, padding=size//2).numpy()                

def compute_local_means(S_fmap, size=1):
    S_fmap_pooled = avg_pool( tiles_to_fmap(S_fmap), size=size )
    return S_fmap_pooled.mean(0, keepdims=True)

def flatten(X_patches):
    return X_patches.reshape(len(X_patches), -1)

def total_pool(X_patches):
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
    
    device = next(eff.parameters()).device
    
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
    
    device = next(eff.parameters()).device
    
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



def auc_global_mean_shift(model, X_in, X_out):
    t0 = time.time()
    S_in = model.transform(np.asarray(X_in), agg="mean")
    S_out = model.transform(np.asarray(X_out), agg="mean")
    
    score_in1 = np.linalg.norm( S_in, axis=1)
    score_out1 = np.linalg.norm( S_out, axis=1)
    t1 = (time.time() - t0)
    
    return roc_auc_score([0] * len(score_in1) + [1] * len(score_out1), np.concatenate([score_in1, score_out1])), t1


def local_mean_shift(model, X, X_in, X_out, size=1):
    S_fmap = model.transform(np.asarray(X), agg="none")
    means = compute_local_means(S_fmap, size=size)
    
    t0 = time.time()
    S_in_fmap = model.transform(np.asarray(X_in), agg="none")
    S_out_fmap = model.transform(np.asarray(X_out), agg="none")
    
    fmap_in_shift = avg_pool( tiles_to_fmap(S_in_fmap), size=size ) - means
    fmap_out_shift = avg_pool( tiles_to_fmap(S_out_fmap), size=size ) - means
    
    score_in = np.linalg.norm(fmap_in_shift.reshape(len(fmap_in_shift), fmap_in_shift.shape[1], -1), axis=1).reshape(len(fmap_in_shift), fmap_in_shift.shape[2], fmap_in_shift.shape[3])
    score_out = np.linalg.norm(fmap_out_shift.reshape(len(fmap_out_shift),fmap_out_shift.shape[1],-1), axis=1).reshape(len(fmap_out_shift), fmap_out_shift.shape[2], fmap_out_shift.shape[3])
    
    return score_in, score_out


def make_anomaly_map(map_i, map_o):
    
    stack = []
    
    mini = np.min([map_i.min(), map_o.min()])
    map_i = map_i - mini
    map_o = map_o - mini
    maxi = np.max([map_i.max(), map_o.max()])
    map_i = map_i / maxi
    map_o = map_o / maxi
    
    for mapi in [map_i, map_o]:
        anomaly_map = mapi[:, None, :, :]
        anomaly_map = batch_resize(anomaly_map, size=(224,224))
        anomaly_map = np.clip(anomaly_map**5, 0, 1)
        zeros = np.zeros_like(anomaly_map)
        anomaly_map = np.concatenate([anomaly_map, zeros, zeros], axis=1)
        stack.append(anomaly_map)

    return stack[0], stack[1]

def auc_local_mean_shift(model, X, X_in, X_out, size=1):
    S_fmap = model.transform(np.asarray(X), agg="none")
    means = compute_local_means(S_fmap, size=size)
    
    t0 = time.time()
    S_in_fmap = model.transform(np.asarray(X_in), agg="none")
    S_out_fmap = model.transform(np.asarray(X_out), agg="none")
    
    fmap_in_shift = avg_pool( tiles_to_fmap(S_in_fmap), size=size ) - means
    fmap_out_shift = avg_pool( tiles_to_fmap(S_out_fmap), size=size ) - means
    
    score_in1 = np.linalg.norm(fmap_in_shift.reshape(len(fmap_in_shift),fmap_in_shift.shape[1],-1), axis=1).max(1)
    score_out1 = np.linalg.norm(fmap_out_shift.reshape(len(fmap_out_shift),fmap_out_shift.shape[1],-1), axis=1).max(1)
    t1 = (time.time() - t0)
    
    return roc_auc_score([0] * len(score_in1) + [1] * len(score_out1), np.concatenate([score_in1, score_out1])), t1



def auc_cluster_mean_shift(model, X, X_in, X_out, use_coreset=True, return_coreset=False, k=None):
    
    S_fmap = model.transform(np.asarray(X), agg="none")
    
    n_tiles = S_fmap.shape[1]
    n_components = S_fmap.shape[-1]
    
    X_ = S_fmap.reshape(-1, n_components)
    
    if k is None: 
        selector = kCenterGreedy(X_, 0, 0)
        selected_idx = selector.select_batch(model=None, already_selected=[], N=int(X_.shape[0]*0.01))
        train_coreset = X_[selected_idx]
        k = len(train_coreset)
 
    n_clusters = k
    print("n_clusters", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=1).fit(X_)

    if use_coreset:
        kmeans.cluster_centers_ = train_coreset
    
    # Patch Clustering
    t0 = time.time()
    S_in_fmap = model.transform(np.asarray(X_in), agg="none")
    S_out_fmap = model.transform(np.asarray(X_out), agg="none")
    
    X_valid_ = S_in_fmap.reshape(-1, n_components)
    X_test_ = S_out_fmap.reshape(-1, n_components)
        
    S_in_assignments = kmeans.predict(X_valid_)
    S_out_assignments = kmeans.predict(X_test_)

    # Compute mean-shift per cluster center
    S_in_means = np.zeros((len(S_in_assignments), n_components))
    S_out_means = np.zeros((len(S_out_assignments), n_components))

    for i in range(n_clusters):
        S_in_means[S_in_assignments == i] += X_valid_[S_in_assignments == i] - kmeans.cluster_centers_[i]
        S_out_means[S_out_assignments == i] += X_test_[S_out_assignments == i] - kmeans.cluster_centers_[i]

    # compute cluster distance for every patch, take maximum!
    scores_valid = np.linalg.norm(S_in_means.reshape(len(S_in_fmap), n_tiles, n_components), axis=2).max(1)
    scores_test = np.linalg.norm(S_out_means.reshape(len(S_out_fmap), n_tiles, n_components), axis=2).max(1)
    t1 = (time.time() - t0)

    if return_coreset:
        return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])) , t1, train_coreset
    
    return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])) , t1      


def auc_sources_mean_shift(model, X, X_in, X_out, k=100):
    
    S = model.transform(np.asarray(X), agg="mean")
    
    n_components = S.shape[-1]
    
    n_clusters = np.min([k, len(S)])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(S)
    
    t0 = time.time()
    S_in = model.transform(np.asarray(X_in), agg="mean")
    S_out = model.transform(np.asarray(X_out), agg="mean")
    
    S_in_assignments = kmeans.predict(S_in)
    S_out_assignments = kmeans.predict(S_out)

    S_in_means = np.zeros_like(S_in)
    S_out_means = np.zeros_like(S_out)
    
    for i in range(n_clusters):

        S_in_means[S_in_assignments == i] += S_in[S_in_assignments == i] - kmeans.cluster_centers_[i]
        S_out_means[S_out_assignments == i] += S_out[S_out_assignments == i] - kmeans.cluster_centers_[i]
        
    scores_valid = np.linalg.norm(S_in_means, axis=1)
    scores_test = np.linalg.norm(S_out_means, axis=1)
    t1 = (time.time() - t0)

    return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])), t1


def auc_coreset(X, X_in, X_out):
    
    X = avg_pool(X, 2)
    X_in = avg_pool(X_in, size=2)
    X_out = avg_pool(X_out, size=2)
    
    n_components = X.shape[1]
    n_tiles =  X.shape[2] * X.shape[3]
    
    X = X.reshape(len(X), n_components, n_tiles) 
    X_in = X_in.reshape(len(X_in), n_components, n_tiles) 
    X_out = X_out.reshape(len(X_out), n_components, n_tiles) 
    
    X = X.transpose(0, 2, 1)
    X_in = X_in.transpose(0, 2, 1)
    X_out = X_out.transpose(0, 2, 1)
    
    selector = kCenterGreedy(X, 0, 0)
    selected_idx = selector.select_batch(model=None, already_selected=[], N=int(X.shape[0]*0.01))
    train_coreset = X[selected_idx]
        
    scores_valid = patch_core_score_2(train_coreset, X_in, n_tiles, b=10, reweight=False) 
    scores_test = patch_core_score_2(train_coreset, X_out, n_tiles, b=10, reweight=False)

    return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test]))        
    