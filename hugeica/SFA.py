import pandas as pd
import torch
from itertools import product
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from hugeica import *
import matplotlib.pyplot as plt

class SFA():

    def  __init__(self, n_components = 30, remove_components = 0, ica=False, shape=(3,32,32), BSZ=(16, 16), stride=4, mode="none", temporal_decorr=False, act=lambda x : x, bs=10000, max_components=100000000):
        assert mode in ["slow", "fast", "none"]

        self.n_components = n_components
        self.max_components = max_components
        self.remove_components = remove_components
        self.ica = ica
        self.shape = shape
        self.BSZ = BSZ
        self.stride = stride
        self.mode = mode
        self.temporal_decorr = temporal_decorr
        self.act = act
        self.dim = self.shape[0]*np.prod(self.BSZ)

        if type(self.n_components) == int and self.n_components == -1:
            self.n_components = self.dim
            
        if type(self.n_components) == float:
            self.n_components = int(self.dim * self.n_components)


    def init_model(self, n_components, bs, max_components):
        if n_components == "kaiser":
            n_components = self.dim

        if n_components > np.min([max_components, bs]):
            print(f"Warning: n_components={n_components} > np.min([max_components={max_components},bs={bs}]). Setting n_components={np.min([max_components, bs])}")
            n_components =  np.min([max_components, bs])

        model = SpatialICA(shape=self.shape, 
                           BSZ=self.BSZ, 
                           padding=0, 
                           stride=self.stride, 
                           n_components=n_components,
                           loss="negexp", 
                           optimistic_whitening_rate=1000, 
                           whitening_strategy="batch", 
                           reduce_lr=True,
                           bs=bs)
        return model
    
    def fit(self, X, epochs = 15, bs=10000, logging=-1, lr=1e-2):
        ####################################
        # Input validation
        ###################################
        if not self.ica and epochs > 1:
            print("Warning: Setting epochs to 1, as self.ica is False")
            epochs = 1

        ####################################
        # TiledICA
        ###################################
        self.model = self.init_model(self.n_components, bs, self.max_components)
        print(f"# Fit SpatialICA({self.n_components}).")
        self.model.fit(X, epochs, X_val=X[:100], logging=logging, lr=lr, bs=bs)
        self.model.cpu()
        
        if not self.ica:
            self.model.net.ica.weight.data = torch.eye(self.model.net.ica.weight.data.shape[0]).to(self.model.device)   
        ####################################
        # Transform
        ###################################
        if self.n_components == "kaiser":
           n_positive_eigvals = kaiser_rule(None, eigvals=self.model.explained_variance_)
           self.n_components = n_positive_eigvals
           self.model = self.init_model(self.n_components, bs, self.max_components)
           print(f"# Re-Fit SpatialICA({self.n_components}).")
           self.model.fit(X, epochs, X_val=X[:100], logging=logging, lr=lr, bs=bs)
           self.model.cpu()
           S = self.model.transform(np.asarray(X), agg="none", act=self.act)
        else:
           S = self.model.transform(np.asarray(X), agg="none", act=self.act)
        
        # Compute some Information measures
        print(f"# Compute ICA metrics.")
        if self.dim > 100*100*3:
            print(f"Warning: Not computing cov(dim={self.dim}).")
            self.H_receptive       =  np.nan
            self.H_signal          =  np.nan
        else:
            self.H_receptive       =  entropy_gaussian(self.model.cov)
            self.H_signal          =  entropy_gaussian(self.model.cov, self.n_components, eigvals=self.model.explained_variance_)
        self.H_signal_white    =  entropy_gaussian(dim=self.n_components)
        self.H_signal_sparse   = -Loss.LogcoshNormalized(torch.FloatTensor(S.reshape(-1, self.n_components))).sum(1).mean(0).numpy()
        self.H_signal_gauss    = -Loss.Gaussian(torch.FloatTensor(S.reshape(-1, self.n_components))).sum(1).mean(0).numpy()

        ####################################
        # SFA 
        ####################################
        S = S.reshape(-1, self.n_components)        
        S_change_score_diff = (S.reshape(-1, self.model.n_tiles, self.n_components) - np.roll(S.reshape(-1, self.model.n_tiles, self.n_components), axis=1, shift=1))
        S_change_score = S_change_score_diff.reshape(len(S), -1)
        
        self.sfa = HugeICA(self.n_components, bs=bs)
        print(f"# Fit SFA({self.n_components}).")
        self.sfa.fit(S_change_score, 1, X_val=S_change_score[:100], logging=logging, lr=1e-2, bs=bs)
        if self.mode == "none":
            slow_idx = np.argsort(self.sfa.var)
            self.T = np.eye(self.n_components).astype(np.float32)[:, slow_idx]
            self.change_variance_ = self.sfa.var[slow_idx]
            self.slow_components = self.n_components
        else:
            self.slow_components = self.n_components - self.remove_components
            slow_idx = np.argsort(self.sfa.explained_variance_)
            if self.mode == "slow":
                self.T = self.sfa.components[:, slow_idx[:self.slow_components]]
                self.change_variance_ = self.sfa.explained_variance_[slow_idx[:self.slow_components]]
            if self.mode == "fast":
                self.T = self.sfa.components[:, slow_idx[-self.slow_components:]]
                self.change_variance_ = self.sfa.explained_variance_[slow_idx[-self.slow_components:]]

        print("# Update the independent components")
        self.model.n_components        = self.slow_components
        self.model.net.ica.weight.data = (self.model.net.ica.components_.T @ torch.from_numpy(self.T).to(self.model.device)).T


        ####################################
        # Transform
        ###################################
        S = self.model.transform(np.asarray(X), agg="none", act=self.act)
        S_diff = (S.reshape(-1, self.model.n_tiles, self.n_components) - np.roll(S.reshape(-1, self.model.n_tiles, self.n_components), axis=1, shift=1))
        S_diff = S_diff.reshape(-1, self.n_components)
        # S_diff = np.diff(S, axis=1).reshape(-1, self.n_components)

        # Compute some Information measures
        def neg_diff(s):
            return -Loss.NegentropyLoss(torch.FloatTensor(s), G_fun=Loss.Logcosh).detach().numpy()

        # Compute some Information measures 
        self.H_neighbor         = entropy_gaussian(self.sfa.cov)
        self.var_diff           = S_change_score_diff.reshape(len(S), -1).var(0).sum() # variance of diffs
        self.var                = S.mean(1).var(0).mean() # variance of output
        self.negH_diff          = neg_diff(S_diff)
        self.CE_gaussian        = -Loss.Gaussian(torch.FloatTensor(S_diff)).sum(1).mean(0).numpy()
        self.d_ruska            = np.abs(self.H_neighbor - self.H_signal_white)
        self.d_ruska_           = self.H_neighbor - self.H_signal_white
        self.KL                 = self.CE_gaussian - self.H_neighbor
        self.H_max              = self.H_receptive/(self.shape[0]*np.prod(self.BSZ)) + (self.H_neighbor/self.n_components) 
        
        ####################################
        # Decorrelation
        ####################################
        if self.temporal_decorr:
            print(f"# Fit Decorr({self.slow_components}).")
            t = S.shape[1]
            self.T_temp = []
            for c in range(self.slow_components):
                klt = HugeICA(t)
                klt.fit(S[:, :, c], 1, X_val=S[:, :, c][:100], logging=-1, lr=1e-2, bs=bs)
                self.T_temp.append(klt)


    @staticmethod
    def hyperparameter_search(X, X_in, X_out, patch_size=[8, 16], n_components=[8, 16], stride=[2, 4], shape=(3,32,32), bs=10000, epochs=1, norm=[1], remove_components=None, logging=-1, max_components=100000000, compute_bpd=True):

        if epochs > 1:
            ica = True
        else:
            ica = False 

        def agg(model, mode, X_in, X_out, ord=norm):
            S_in = model.transform(np.asarray(X_in), agg=mode)
            S_out = model.transform(np.asarray(X_out), agg=mode)
            if ( np.isnan(np.linalg.norm(S_in, axis=1, ord=ord)).sum() + np.isnan(np.linalg.norm(S_out, axis=1, ord=ord)).sum()) > 0:
                auc = 0.
            elif ( np.isinf(np.linalg.norm(S_in, axis=1, ord=ord)).sum() + np.isinf(np.linalg.norm(S_out, axis=1, ord=ord)).sum()) > 0:
                auc = 0.
            else:
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([np.linalg.norm(S_in, axis=1, ord=ord), np.linalg.norm(S_out, axis=1, ord=ord)]))
            return auc

        def lhd(model, X_in, X_out, mean, std):
            # print(X_in.shape, X_out.shape)
            ins_lhd = bpd_pca_elbo(model, X_in, mean, std)
            outs_lhd = bpd_pca_elbo(model, X_out, mean, std)
            auc = roc_auc_score([0] * len(X_in) + [1] * len(X_out), np.concatenate([ins_lhd, outs_lhd]))
            return auc, ins_lhd.mean()
        
        def preprocess(X, X_in, X_out):
            X_, _ = dequantize(X) 
            X_, _ = to_norm_contrast(X_)
            # mean, std = X_.mean(0), X_.std(0)
            mean, std = np.zeros(X_.mean(0).shape), X_.std()

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
                if s == None: # set to patch_size
                    s = p
                for c in n_components:
                    #if type(c) == str or c <= p*p*shape[0]:
                    try:
                        model = SFA(shape=shape, 
                                        BSZ=(p, p), 
                                        stride=s, 
                                        n_components=c,
                                        remove_components=remove_components,
                                        ica=ica,
                                        bs=bs,
                                        max_components=max_components)
                        model.fit(X_, epochs, bs=bs, logging=logging)
                        S = model.transform(np.asarray(X), agg="sum")
                        for nor in norm:
                            auc          = [agg(model, mode, X_in_, X_out_, nor) for mode in ["var", "sum", "mean"]]
                            bpd_field = auc_lhd = bpd = 0
                            if compute_bpd:
                                bpd_field    = bpd_pca_elbo_receptive(model.model, X_in, mean, std).mean()
                                auc_lhd, bpd = lhd(model, X_in, X_out, mean, std)
                            spread = model.change_variance_.max() - model.change_variance_.min()
                            k_min  = model.change_variance_.min()
                            k_max  = model.change_variance_.max()
                            k      = model.change_variance_.max()/model.change_variance_.min()
                            H_receptive  = model.H_receptive
                            H_neighbor   = model.H_neighbor
                            CE_gaussian  = model.CE_gaussian
                            H_signal        = model.H_signal
                            H_signal_white  = model.H_signal_white
                            H_signal_sparse = model.H_signal_sparse
                            H_signal_gauss  = model.H_signal_gauss
                            negH_diff    = model.negH_diff
                            KL           = model.KL
                            H_max        = model.H_max
                            d_ruska      = model.d_ruska
                            d_ruska_      = model.d_ruska_
                            var_diff     = model.var_diff
                            var_sum      = model.var
                            bookkeeping.append([p, s, model.n_components, nor, \
                                k_min, k_max, k, bpd, bpd_field, \
                                    H_receptive, H_signal, H_signal_white, H_neighbor, H_signal_sparse, H_signal_gauss, CE_gaussian, \
                                    var_diff, var_sum, KL, d_ruska, d_ruska_, negH_diff, H_max, auc_lhd] + auc )
                    except:
                        e = sys.exc_info()[0]
                        v = sys.exc_info()[1]
                        print("Error in config:", p, s, c, "- caused by ", e, v)
                        
        bookkeeping = np.asarray(bookkeeping).astype(np.float32)
        return pd.DataFrame(bookkeeping, columns=["patch_size", "s", "n_components", "nor", \
                                    "k_min", "k_max", "k", "bpd", "bpd_field", \
                                        "H_receptive", "H_signal", "H_signal_white", "H_neighbor", "H_signal_sparse", "H_signal_gauss", "CE_gaussian", \
                                            "var_diff", "var_sum", "KL", "d_ruska", "d_ruska_", "negH_diff", "H_max", "lhd" ,"var", "sum", "mean"])       
         

    
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
            S = self.model.transform(X, exponent=exponent, agg="none", act=act)
            for c in range(self.slow_components):
                klt = self.T_temp[c]
                S[:, :, c] = klt.transform(S[:, :, c]) 
            S = self.model.agg(S, agg)
        else:
            S = self.model.transform(X, exponent=exponent, agg=agg, act=act)
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