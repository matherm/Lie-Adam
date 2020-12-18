from itertools import product
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from hugeica import *
    
def neg_diff(s):
    return -Loss.NegentropyLoss(torch.FloatTensor(s).unsqueeze(1), G_fun=Loss.Logcosh).numpy()

class SFA():

    def  __init__(self, n_components = 30, slow_components = 2, squared_change=False, ica=False, shape=(3,32,32), addition=False, BSZ=(16, 16), stride=4, mode="slow", temporal_decorr=False, act=lambda x : x):
        assert mode in ["slow", "fast", "none"]

        self.n_components = n_components
        self.slow_components = slow_components
        self.squared_change = squared_change
        self.ica = ica
        self.shape = shape
        self.addition = False
        self.BSZ = BSZ
        self.stride = stride
        self.mode = mode
        self.temporal_decorr = temporal_decorr
        self.act = act
        
    
    def fit(self, X, epochs = 15, bs=10000, logging=-1, lr=1e-2):
        if not self.ica and epochs > 1:
            print("Setting epochs to 1, as self.ica is False")
            epochs = 1

        ####################################
        # TiledICA
        ###################################
        self.model = SpatialICA(shape=self.shape, 
                           BSZ=self.BSZ, 
                           padding=0, 
                           stride=self.stride, 
                           n_components=self.n_components,
                           loss="negexp", 
                           optimistic_whitening_rate=1000, 
                           whitening_strategy="batch", 
                           reduce_lr=True)
        self.model.fit(X, epochs, X_val=X[:100], logging=logging, lr=lr, bs=bs)
        self.model.cpu()
        
        if not self.ica:
            self.model.net.ica.weight.data = torch.eye(self.model.net.ica.weight.data.shape[0]).to(self.model.device)   
        ####################################
        # Transform
        ###################################
        S = self.model.transform(X, agg="none", bs=bs, act=self.act)
        
        # Compute some Information measures
        self.H_receptive       =  entropy_gaussian(self.model.cov)
        self.H_signal_sparse   = -Loss.LogcoshNormalized(torch.FloatTensor(S.reshape(-1, self.n_components))).sum(1).mean(0).numpy()
        self.H_signal_gauss    = -Loss.Gaussian(torch.FloatTensor(S.reshape(-1, self.n_components))).sum(1).mean(0).numpy()
        self.I_signal          =  np.asarray([entropy_bins( np.tanh(S.reshape(-1, self.n_components)[:, i]) ) for i in range(self.n_components)]).sum() -  entropy_gaussian(np.eye(self.n_components))
        self.H_signal2d_sparse = -Loss.LogcoshNormalized(torch.FloatTensor(S)).sum((1,2)).mean(0).numpy()
        self.H_signal2d_gauss  = -Loss.Gaussian(torch.FloatTensor(S)).sum((1,2)).mean(0).numpy()
        self.I_signal2d        =  np.asarray([entropy_bins( np.tanh(S[:,i,j]) ) for i,j in product(range(S.shape[1]),range(self.n_components))]).sum() -  entropy_gaussian(dim=self.n_components*S.shape[1])

        ####################################
        # SFA 
        ####################################
        S = S.reshape(-1, self.n_components)        
        if self.addition:
            S_change_score = (S.reshape(-1, self.model.n_tiles, self.n_components) + np.roll(S.reshape(-1, self.model.n_tiles, self.n_components), axis=1, shift=1))
        else:
            S_change_score = (S.reshape(-1, self.model.n_tiles, self.n_components) - np.roll(S.reshape(-1, self.model.n_tiles, self.n_components), axis=1, shift=1))
        S_change_score = S_change_score.reshape(len(S), -1)

        if self.squared_change:
            S_change_score = S_change_score**2
          
        self.sfa = HugeICA(self.n_components)
        self.sfa.fit(S_change_score, 1, X_val=S_change_score[:100], logging=-1, lr=1e-2, bs=bs)
        if self.mode == "none":
            assert self.slow_components == self.n_components
            slow_idx = np.argsort(self.sfa.var)
            self.T = np.eye(self.n_components).astype(np.float32)[:, slow_idx]
            self.change_variance_ = self.sfa.var[slow_idx]
        else:
            slow_idx = np.argsort(self.sfa.explained_variance_)
            if self.mode == "slow":
                self.T = self.sfa.components[:, slow_idx[:self.slow_components]]
                self.change_variance_ = self.sfa.explained_variance_[slow_idx[:self.slow_components]]
            if self.mode == "fast":
                self.T = self.sfa.components[:, slow_idx[-self.slow_components:]]
                self.change_variance_ = self.sfa.explained_variance_[slow_idx[-self.slow_components:]]

        # Update the indepenendent components
        self.model.n_components        = self.slow_components
        self.model.net.ica.weight.data =  (self.model.net.ica.components_.T @ torch.from_numpy(self.T).to(self.model.device)).T

        # Compute some Information measures
        self.H_neighbor   = entropy_gaussian(self.sfa.cov)
        self.I_neighbor   = self.H_neighbor - entropy_gaussian(dim=self.n_components)
        self.H_receptive2 = self.H_receptive
        self.H_max        =  self.H_receptive/(3*np.prod(self.BSZ)) + (self.H_neighbor/self.n_components) 
        
        ####################################
        # Transform
        ###################################
        S = self.model.transform(X, agg="none", bs=bs, act=self.act)
        
        # Compute some Information measures
        self.I_neighbor_negentropy   = neg_diff(np.abs(np.diff(S, axis=1).flatten()))

        ####################################
        # Decorrelation
        ####################################
        if self.temporal_decorr:
            t = S.shape[1]
            self.T_temp = []
            for c in range(self.slow_components):
                klt = HugeICA(t)
                klt.fit(S[:, :, c], 1, X_val=S[:, :, c][:100], logging=-1, lr=1e-2, bs=bs)
                self.T_temp.append(klt)


    @staticmethod
    def hyperparameter_search(X, X_in, X_out, patch_size=[8, 16], n_components=[8, 16], stride=[2, 4], shape=(3,32,32), bs=10000, epochs=1, norm=[1], remove_components=None):

        if epochs > 1:
            ica = True
        else:
            ica = False 

        def agg(mode, X_in, X_out, ord=norm):
            S_in = model.transform(X_in, agg=mode)
            S_out = model.transform(X_out, agg=mode)
            if ( np.isnan(np.linalg.norm(S_in, axis=1, ord=ord)).sum() + np.isnan(np.linalg.norm(S_out, axis=1, ord=ord)).sum()) > 0:
                auc = 0.
            elif ( np.isinf(np.linalg.norm(S_in, axis=1, ord=ord)).sum() + np.isinf(np.linalg.norm(S_out, axis=1, ord=ord)).sum()) > 0:
                auc = 0.
            else:
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([np.linalg.norm(S_in, axis=1, ord=ord), np.linalg.norm(S_out, axis=1, ord=ord)]))
            return auc
        
        def preprocess(X, X_in, X_out):
            X_, _ = dequantize(X)
            X_, _ = to_norm_contrast(X_)
            mean, std = X_.mean(), X_.std()    

            X_, _ = scale(X_, mean, std)

            X_in_, _ = dequantize(X_in)
            X_in_, _ = to_norm_contrast(X_in_)
            X_in_, _ = scale(X_in_, mean, std)

            X_out_, _ = dequantize(X_out)
            X_out_, _ = to_norm_contrast(X_out_)
            X_out_, _ = scale(X_out_, mean, std)
            return X_, X_in_, X_out_, mean, std
        
        X_, X_in_, X_out_, mean, std = preprocess(X, X_in, X_out)
        
        bookkeeping = []
        for p in patch_size:
            for s in stride:
                for c in n_components:
                    if c <= p*p*shape[0]:
                        try:
                            model = SFA(shape=shape, 
                                            BSZ=(p, p), 
                                            stride=s, 
                                            n_components=c,
                                            slow_components=c if remove_components is None else c - remove_components,
                                            ica = ica)
                            model.fit(X_, epochs, bs=bs, logging=-1)

                            for nor in norm:
                                auc    = [agg(mode, X_in_, X_out_, nor) for mode in ["var", "std", "sum", "mean", "prod", "invsum", "max", "min"]]
                                bbc    = bits_back_code(model, X, mean, std).mean()
                                bbcpd  = bits_back_code_per_dim(model, X, mean, std).mean()
                                bpd    = bpd_pca_elbo(model, X, mean, std).mean()
                                spread = model.change_variance_.max() - model.change_variance_.min()
                                k_min  = model.change_variance_.min()
                                k_max  = model.change_variance_.max()
                                k      = model.change_variance_.max()/model.change_variance_.min()
                                K      = kurt( model.transform(X, agg="sum")).mean()
                                H_receptive  = model.H_receptive
                                H_receptive2 = model.H_receptive2
                                H_neighbor   = model.H_neighbor
                                H_signal_sparse = model.H_signal_sparse
                                H_signal_gauss  = model.H_signal_gauss
                                I_signal             = model.I_signal
                                I_neighbor           = model.I_neighbor
                                I_neighbor_negentropy= model.I_neighbor_negentropy
                                H_max                = model.H_max
                                bookkeeping.append([p, s, c, nor, \
                                    spread, k_min, k_max, k, K, bbc, bpd, bbcpd, \
                                        H_receptive, H_receptive2, H_neighbor, H_signal_sparse, H_signal_gauss, \
                                            I_signal, I_neighbor, I_neighbor_negentropy, H_max] + auc )
                        except:
                            e = sys.exc_info()[0]
                            print("Error in config:", p, s, c, nor, "- caused by ", e)
                            
        return np.asarray(bookkeeping).astype(np.float32)
    
    def to(self, device):
        self.model.to(device)

    def elbo(self, *args, **kwargs):
        return self.model.elbo(*args, **kwargs)

    def entropy(self, *args, **kwargs):
        return self.model.entropy(*args, **kwargs)

    def cross_entropy(self, *args, **kwargs):
        return self.model.cross_entropy(*args, **kwargs)

    def cross_entropy_x(self, *args, **kwargs):
        return self.model.cross_entropy_x(*args, **kwargs)

    def bits_back_code(self, *args, **kwargs):
        return self.model.bits_back_code(*args, **kwargs)

    def reconstruct(self, *args, **kwargs):
        return self.model.reconstruct(*args, **kwargs)

    def transform(self, X, exponent=1, agg="none", bs=-1, act=lambda x : x):
        if self.temporal_decorr:
            S = self.model.transform(X, exponent=exponent, agg="none", bs=bs, act=act)
            for c in range(self.slow_components):
                klt = self.T_temp[c]
                S[:, :, c] = klt.transform(S[:, :, c]) 
            S = self.model.agg(S, agg)
        else:
            S = self.model.transform(X, exponent=exponent, agg=agg, bs=bs, act=act)
        return S

    def martingale(self, X):
        """
        https://www.researchgate.net/publication/267629210_Testing_the_Martingale_Hypothesis
        """
        slope, inter = batch_regression(X)
        return inter 
    
    def score(self, X, ord=2, abs=False, exp=1):
        if abs:
            score = np.abs(self.transform(X))
        elif exp > 1:
            score = self.transform(X)**exp
        else:
            score = self.transform(X)
        score = np.linalg.norm(score.sum(1), axis=1, ord=ord)
        assert len(score) == len(X)
        return score


class TAM():

    def __init__(self,  n_components = 30, ica=False, shape=(3,32,32), BSZ=(16, 16), stride=4):
        self.n_components = n_components
        self.ica = ica
        self.shape = shape
        self.BSZ = BSZ
        self.stride = stride
    
    def fit(self, X, epochs = 15):
        if not self.ica:
            epochs = 1
        self.model = SpatialICA(shape=self.shape, 
                           BSZ=self.BSZ, 
                           padding=0, 
                           stride=self.stride, 
                           n_components=self.n_components,
                           loss="negexp", 
                           optimistic_whitening_rate=1000, 
                           whitening_strategy="batch", 
                           reduce_lr=True)
        self.model.fit(X, 1, X_val=X[:100], logging=-1, lr=1e-2, bs=10000)
        
        if not self.ica:
            self.model.net.ica.weight.data = torch.eye(self.model.net.ica.weight.data.shape[0]).to(self.model.device)   

    def elbo(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def transform(self, X, agg="none"):
        S = self.model.transform(X, agg=agg)
        return S
    
    def score(self, X, ord=1):
        score = np.linalg.norm(self.transform(X).sum(1), axis=1, ord=0.5)
        assert len(score) == len(X)
        return score