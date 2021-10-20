import pandas as pd
import torch
from itertools import product
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from hugeica import *
import matplotlib.pyplot as plt

# Compute some Information measures
def negH(s, avg=False, reduce=True):
    if reduce:
        if avg:
            return -Loss.NegentropyLoss(torch.FloatTensor(s), G_fun=Loss.Logcosh, bootstraps=20).detach().numpy() / s.shape[1]
        else:
            return -Loss.NegentropyLoss(torch.FloatTensor(s), G_fun=Loss.Logcosh, bootstraps=20).detach().numpy()
    else:
        return Loss.NegentropyResample(torch.FloatTensor(s), G_fun=Loss.Logcosh, bootstraps=20).detach().numpy()

class SFA():

    def  __init__(self, n_components = 30, remove_components = 0, shape=(3,32,32), BSZ=(16, 16), stride=4, mode="none", temporal_decorr=False, act=lambda x : x,  max_components=100000000,  use_conv=False, inter_image_diffs=True, extended_entropies=False):
        assert mode in ["slow", "fast", "sparse", "none", "slow_null", "random", "ta", "ica"]

        self.n_components = n_components
        self.max_components = max_components
        self.remove_components = remove_components
        self.shape = shape
        self.use_conv = use_conv
        self.inter_image_diffs = inter_image_diffs
        self.BSZ = BSZ
        self.n_tiles =  ((shape[1] - BSZ[0]) // stride + 1)**2
        self.stride = stride
        self.mode = mode
        self.temporal_decorr = temporal_decorr
        self.act = act
        self.dim = self.shape[0]*np.prod(self.BSZ)
        self.extended_entropies = extended_entropies

        if type(self.n_components) == int and self.n_components == -1:
            self.n_components = self.dim
            
        if type(self.n_components) == float:
            self.n_components = int(self.dim * self.n_components)

    
    def init_model(self, n_components, bs, max_components, conv=False):

        if conv:
            bs = bs * self.n_tiles

        if n_components == "kaiser" or n_components == "mle" or n_components == "q90" or n_components == "q95":
            n_components = self.dim

        if n_components > np.min([max_components, bs]):
            print(f"Warning: n_components={n_components} > np.min([max_components={max_components},bs={bs}]). Setting n_components={np.min([max_components, bs])}")
            n_components =  np.min([max_components, bs])

        if conv:     
            model = ConvSpatialICA(shape=self.shape, 
                            BSZ=self.BSZ, 
                            padding=0, 
                            stride=self.stride, 
                            n_components=n_components,
                            loss= (lambda x : -Loss.Exp(x.view(-1, self.n_tiles, x.size(1)).mean(1))) if self.mode == "ta" else "negexp", 
                            optimistic_whitening_rate=1000, 
                            whitening_strategy="batch", 
                            reduce_lr=True,
                            bs=bs,
                            init_eye=False if self.mode == "ica" else True)
        else:
            model = SpatialICA(shape=self.shape, 
                            BSZ=self.BSZ, 
                            padding=0, 
                            stride=self.stride, 
                            n_components=n_components,
                            loss=  (lambda x : -Loss.Exp(x.view(-1, self.n_tiles, x.size(1)).mean(1))) if self.mode == "ta" else "negexp", 
                            optimistic_whitening_rate=1000, 
                            whitening_strategy="batch", 
                            reduce_lr=True,
                            bs=bs,
                            init_eye=False if self.mode == "ica" else True)
        return model
    
    def fit(self, X, epochs = 15, bs=10000, logging=-1, lr=1e-2, resample=False):
        ####################################
        # Input Validation
        ###################################

        if self.use_conv == True:
            if bs % self.n_tiles > 0:
                bs_ = bs
                bs = bs - bs % self.n_tiles
                print(f"Warning: bs={bs_} is not a multiple of n_tiles={self.n_tiles}. Setting bs={bs}")
            bs = bs // self.n_tiles

        elif self.mode == "ta":
            if bs % self.n_tiles > 0:
                bs_ = bs
                bs = bs - bs % self.n_tiles
                print(f"Warning: bs={bs_} is not a multiple of n_tiles={self.n_tiles}. Setting bs={bs}")



        ####################################
        # TiledICA
        ###################################
        print(f"# Fit SpatialICA({self.n_components}).")
        epochs_ = epochs
        if self.n_components == "kaiser" or self.n_components == "mle" or self.n_components == "q90" or self.n_components == "q95":
            epochs_ = 1

        self.model = self.init_model(self.n_components, bs, self.max_components, self.use_conv)
        self.model.fit(X, epochs_, X_val=X[:100], logging=logging, lr=lr, bs=bs, resample=resample)
        self.model.cpu()
        
        #if not self.ica:
            #self.model.net.ica.weight.data = torch.eye(self.model.net.ica.weight.data.shape[0]).to(self.model.device)   
        ####################################
        # Transform
        ###################################
        if self.n_components == "kaiser" or self.n_components == "mle" or self.n_components == "q90" or self.n_components == "q95":
            self.original_explained_variance_ = self.model.explained_variance_
            if self.n_components == "kaiser":
                self.n_components = kaiser_rule(None, eigvals=self.model.explained_variance_)  
            elif self.n_components == "mle":
                self.n_components = mle_rule(self.model.explained_variance_, len(X))
            elif self.n_components == "q90":
                self.n_components = quantile_rule(None, eigvals=self.model.explained_variance_, explained_variance=0.90)
            elif self.n_components == "q95":
                self.n_components = quantile_rule(None, eigvals=self.model.explained_variance_, explained_variance=0.95)
            
            print(f"# Re-Fit SpatialICA({self.n_components}).")
            self.model = self.init_model(self.n_components, bs, self.max_components, self.use_conv)
            self.model.fit(X, epochs_, X_val=X[:100], logging=logging, lr=lr, bs=bs, resample=resample)
            self.model.cpu()
            S = self.model.transform(np.asarray(X), agg="none", act=self.act)
        else:
            self.n_components = self.model.n_components
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
        # SFA / ICA
        ####################################
        S = S.reshape(-1, self.n_components)        
        print(f"# Fit SFA({self.n_components}).")
        S_change_score = (S.reshape(-1, self.model.n_tiles, self.n_components) - np.roll(S.reshape(-1, self.model.n_tiles, self.n_components), axis=1, shift=1))
        S_change_score = S_change_score.reshape(len(S), -1)
        if self.inter_image_diffs == False:
            invalid_diffs = ((np.arange(len(X)) + 1) * self.model.n_tiles) - 1
            S_change_score = np.delete(S_change_score, invalid_diffs, axis=0)
        
        self.sfa = HugeICA(self.n_components, bs=bs)
        if self.mode == "slow" or self.mode == "slow_null" or self.mode == "fast":
            self.sfa.fit(S_change_score, 1, X_val=S_change_score[:100], logging=logging, lr=1e-2, bs=bs)
            self.change_variance_ = self.sfa.var
            self.sfa_cov = self.sfa.cov
        else:
            self.change_variance_ = np.zeros(self.n_components)
            self.sfa_cov = np.eye(self.n_components)


        if self.remove_components == "q95":
            self.reduced_components = np.min([quantile_rule(None, eigvals=self.original_explained_variance_, explained_variance=0.95), self.n_components])
            self.remove_components = self.n_components - self.reduced_components
        elif self.remove_components == "q90":
            self.reduced_components = np.min([quantile_rule(None, eigvals=self.original_explained_variance_, explained_variance=0.90), self.n_components])
            self.remove_components = self.n_components - self.reduced_components
        else:
            self.reduced_components = self.n_components - self.remove_components

        if self.extended_entropies:
            print(f"# Fit Add-SFA({self.n_components}).")
            S_change_score = (S.reshape(-1, self.model.n_tiles, self.n_components) + np.roll(S.reshape(-1, self.model.n_tiles, self.n_components), axis=1, shift=1))
            S_change_score = S_change_score.reshape(len(S), -1)
            if self.inter_image_diffs == False:
                invalid_diffs = ((np.arange(len(X)) + 1) * self.model.n_tiles) - 1
                S_change_score = np.delete(S_change_score, invalid_diffs, axis=0)
            sfa_add = HugeICA(self.n_components, bs=bs)
            sfa_add.fit(S_change_score, 1, X_val=S_change_score[:100], logging=logging, lr=1e-2, bs=bs)
        
        if self.mode == "slow":
            print("# Update the independent components")
            slow_idx = np.argsort(self.sfa.explained_variance_)
            self.T = self.sfa.components[:, slow_idx[:self.reduced_components]]
            self.change_variance_ = self.sfa.explained_variance_[slow_idx[:self.reduced_components]]
            self.model.n_components        = self.reduced_components
            self.model.net.ica.weight.data = (self.model.net.ica.components_.T @ torch.from_numpy(self.T).to(self.model.device)).T
        elif self.mode == "slow_null":
            print("# Update the independent components")
            slow_idx = np.argsort(self.sfa.explained_variance_)
            self.T = self.sfa.components[:, slow_idx]
            self.T[:, self.reduced_components:] = 0.
            self.change_variance_ = self.sfa.explained_variance_[slow_idx]
            self.change_variance_[self.reduced_components:] = 0.
            self.reduced_components =  self.model.n_components
            self.model.net.ica.weight.data = (self.model.net.ica.components_.T @ torch.from_numpy(self.T).to(self.model.device)).T
        elif self.mode == "fast":
            print("# Update the independent components")
            slow_idx = np.argsort(self.sfa.explained_variance_)
            self.T = self.sfa.components[:, slow_idx[-self.reduced_components:]]
            self.change_variance_ = self.sfa.explained_variance_[slow_idx[-self.reduced_components:]]
            self.model.n_components        = self.reduced_components
            self.model.net.ica.weight.data = (self.model.net.ica.components_.T @ torch.from_numpy(self.T).to(self.model.device)).T
        elif self.mode == "sparse":
            S = self.model.transform(np.asarray(X), agg="none", act=self.act)
            sparse_idx = np.argsort(negH(S.reshape(-1, S.shape[2]), avg=False, reduce=False))
            print("# Update the independent components")
            self.model.n_components        = self.reduced_components
            self.T = self.model.net.ica.components_[sparse_idx[-self.reduced_components:]]
            self.model.net.ica.weight.data = self.T
        elif self.mode == "random":
            random_idx = np.random.permutation(self.n_components)
            print("# Update the independent components")
            self.model.n_components        = self.reduced_components
            self.T = self.model.net.ica.components_[random_idx[-self.reduced_components:]]
            self.model.net.ica.weight.data = self.T
        elif self.mode == "none":
            print("# Update the independent components")
            self.model.net.ica.weight.data = torch.eye(self.model.n_components).to(self.model.device)
        else:
            self.reduced_components = self.n_components

        
        ####################################
        # Transform
        ###################################
        print("# Compute information measures")
        S = self.model.transform(np.asarray(X), agg="none", act=self.act)
        S_diff = (S.reshape(-1, self.model.n_tiles, self.reduced_components) - np.roll(S.reshape(-1, self.model.n_tiles, self.reduced_components), axis=1, shift=1))
        S_add = (S.reshape(-1, self.model.n_tiles, self.reduced_components) + np.roll(S.reshape(-1, self.model.n_tiles, self.reduced_components), axis=1, shift=1))
        S_diff = S_diff.reshape(-1, self.reduced_components)
        S_add = S_add.reshape(-1, self.reduced_components)
        if self.inter_image_diffs == False:
            invalid_diffs = ((np.arange(len(X)) + 1) * self.model.n_tiles) - 1
            S_diff = np.delete(S_diff, invalid_diffs, axis=0)


        # Compute some Information measures
        self.negH_sum           =  negH(S.mean(1), avg=False)
        self.H_neighbor         =  entropy_gaussian(self.sfa_cov)
        self.var_diff           =  S_diff.reshape(len(S), -1).var(0).mean() # variance of diffs
        self.var                =  S.mean(1).var(0).mean() # variance of output
        self.local_var          =  S.var(1).mean() # variance of output
        self.H_output           =  entropy_gaussian(np.cov(S.mean(1).T))
        self.H_joint            =  entropy_gaussian(np.cov(S.reshape(-1, self.reduced_components)[:2*int((len(S)*self.model.n_tiles)/2)].reshape(-1, 2 * self.reduced_components).T)) if self.extended_entropies else -1
        self.H_cond             =  entropy_conditional(S.reshape(-1, self.reduced_components), S_add) if self.extended_entropies else -1
        self.negH_diff          =  negH(S_diff, avg=False)
        self.negH_diff_avg      =  negH(S_diff, avg=True)
        self.kurt               =  Loss.K(S_diff).mean()
        self.CE_gaussian        = -Loss.Gaussian(torch.FloatTensor(S_diff)).sum(1).mean(0).numpy()
        self.d_ruzsa            =  np.abs(self.H_neighbor - self.H_signal_white)
        self.d_ruzsa_add        =  np.abs(entropy_gaussian(sfa_add.cov) - self.H_signal_white) if self.extended_entropies else -1
        self.d_ruzsa_output     =  np.abs(self.H_output - self.H_signal_white)
        self.d_ruzsa_           =  self.H_neighbor - self.H_signal_white
        self.KL                 =  self.CE_gaussian - self.H_neighbor
        self.H_max              =  self.H_receptive/(self.shape[0]*np.prod(self.BSZ)) + (self.H_neighbor/self.reduced_components) 
          
        
        ####################################
        # Decorrelation
        ####################################
        if self.temporal_decorr:
            print(f"# Fit Decorr({self.reduced_components}).")
            t = S.shape[1]
            self.T_temp = []
            for c in range(self.reduced_components):
                klt = HugeICA(t)
                klt.fit(S[:, :, c], 1, X_val=S[:, :, c][:100], logging=-1, lr=1e-2, bs=bs)
                self.T_temp.append(klt)


    @staticmethod
    def hyperparameter_search(X, X_in, X_out, patch_size=[8, 16], n_components=[8, 16], stride=[2, 4], shape=(3,32,32), 
                                bs=10000, lr=1e-2, epochs=1, norm=[1], remove_components=[0], logging=-1, max_components=100000000, 
                                compute_bpd=True, mode="none", use_conv=False, norm_contrast=True, DC=True, channels=None, inter_image_diffs=True, 
                                extended_entropies=False, aucs=["var", "sum", "mean", "hotelling", "martingale", "mean_shift", "typicality", "avg_patch_reconstruct"]):


        def agg(model, mode, X_in, X_out, ord=norm):
            if mode == "hotelling":
                S_in = model.hotelling(X_in)
                S_out = model.hotelling(X_out)
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([S_in, S_out]))
            elif mode == "martingale":
                S_in = model.martingale( model.transform(np.asarray(X_in), agg="none") )
                S_out = model.martingale( model.transform(np.asarray(X_out), agg="none"))
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([np.linalg.norm(S_in, axis=1, ord=1), np.linalg.norm(S_out, axis=1, ord=1)]))
            elif mode == "typicality":
                S_in = -model.typical_set(np.asarray(X_in))
                S_out = -model.typical_set(np.asarray(X_out))
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([S_in, S_out]))
            elif mode == "mean_shift":
                S_in = model.mean_shift(np.asarray(X_in))
                S_out = model.mean_shift(np.asarray(X_out))
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([np.linalg.norm(S_in, axis=1, ord=2), np.linalg.norm(S_out, axis=1, ord=2)]))
            elif mode == "avg_patch_reconstruct":
                S_in = model.avg_patch_reconstruct(np.asarray(X_in))
                S_out = model.avg_patch_reconstruct(np.asarray(X_out))
                auc = roc_auc_score([0] * len(S_in) + [1] * len(S_out), np.concatenate([S_in, S_out]))
            else:
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
            if norm_contrast:
                X_, _ = to_norm_contrast(X_, DC=DC, channels=channels)
            # mean, std = X_.mean(0), X_.std(0)
            mean, std = np.zeros(X_.mean(0).shape), X_.std()
            X_, _ = scale(X_, mean, std)

            X_in_, _ = dequantize(X_in)
            if norm_contrast:
                X_in_, _ = to_norm_contrast(X_in_, DC=DC, channels=channels)
            X_in_, _ = scale(X_in_, mean, std)

            X_out_, _ = dequantize(X_out)
            if norm_contrast:
                X_out_, _ = to_norm_contrast(X_out_, DC=DC, channels=channels)
            X_out_, _ = scale(X_out_, mean, std)
            
            return X_, X_in_, X_out_, mean, std
        
        X_, X_in_, X_out_, mean, std = preprocess(X, X_in, X_out)
        
        bookkeeping = []
        for p in patch_size:
            for s in stride:
                if s == None: # set to patch_size
                    s = p
                for c in n_components:
                    for r in remove_components:
                        try:
                            model = SFA(shape=shape, 
                                            BSZ=(p, p), 
                                            stride=s, 
                                            n_components=c,
                                            remove_components=r,
                                            max_components=max_components,
                                            mode=mode,
                                            use_conv=use_conv,
                                            inter_image_diffs=inter_image_diffs,
                                            extended_entropies=extended_entropies)
                            model.fit(X_, epochs, bs=bs, lr=lr, logging=logging)
                            for nor in norm:
                                print("# Compute AUCs")
                                auc          = [agg(model, mode, X_in_, X_out_, nor) for mode in aucs]
                                bpd_field = auc_lhd = bpd = 0
                                if compute_bpd:
                                    bpd_field    = bpd_pca_elbo_receptive(model.model, X_in, mean, std).mean()
                                    auc_lhd, bpd = lhd(model, X_in, X_out, mean, std)
                                spread = model.change_variance_.max() - model.change_variance_.min()
                                k_min  = model.change_variance_.min()
                                k_max  = model.change_variance_.max()
                                k      = model.change_variance_.max()/model.change_variance_.min()
                                kurt   = model.kurt
                                H_receptive  = model.H_receptive
                                H_neighbor   = model.H_neighbor
                                CE_gaussian  = model.CE_gaussian
                                H_signal        = model.H_signal
                                H_signal_white  = model.H_signal_white
                                H_signal_sparse = model.H_signal_sparse
                                H_signal_gauss  = model.H_signal_gauss
                                H_joint      = model.H_joint
                                H_cond       = model.H_cond
                                negH_diff    = model.negH_diff
                                negH_diff_avg= model.negH_diff_avg
                                negH_sum     = model.negH_sum
                                KL           = model.KL
                                H_max        = model.H_max
                                d_ruzsa_add  = model.d_ruzsa_add
                                d_ruzsa      = model.d_ruzsa
                                d_ruzsa_     = model.d_ruzsa_
                                var_diff     = model.var_diff
                                var_sum      = model.var
                                local_var    = model.local_var
                                total_var    = model.var * model.reduced_components
                                H_output     = model.H_output
                                bookkeeping.append([p, s, model.reduced_components, nor, model.remove_components, \
                                    k_min, k_max, k, kurt, bpd, bpd_field, \
                                        H_receptive, H_signal, H_signal_white, H_neighbor, H_signal_sparse, H_signal_gauss, H_joint, H_cond, CE_gaussian, \
                                        var_diff, var_sum, local_var, total_var, H_output, KL, d_ruzsa_add, d_ruzsa, d_ruzsa_, negH_diff, negH_diff_avg, negH_sum, H_max, auc_lhd] + auc )
                        except NotImplementedError:
                            e = sys.exc_info()[0]
                            v = sys.exc_info()[1]
                            print("Error in config:", p, s, c, "- caused by ", e, v)
                        
        bookkeeping = np.asarray(bookkeeping).astype(np.float32)
        hyp = pd.DataFrame(bookkeeping, columns=["patch_size", "s", "n_components", "nor", "remove_components", \
                                    "k_min", "k_max", "k", "kurt", "bpd", "bpd_field", \
                                        "H_receptive", "H_signal", "H_signal_white", "H_neighbor", "H_signal_sparse", "H_signal_gauss", "H_joint", "H_cond", "CE_gaussian", \
                                            "var_diff", "var_sum", "local_var", "total_var", "H_output", "KL", "d_ruzsa_add", "d_ruzsa", "d_ruzsa_", "negH_diff", "negH_diff_avg", "negH_sum", "H_max", "lhd"] + aucs)       
        
        return hyp
         

    
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

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def transform(self, X, exponent=1, agg="none", bs=-1, act=lambda x : x, resample=False):
        if self.temporal_decorr:
            S = self.model.transform(X, exponent=exponent, agg="none", act=act, resample=resample)
            for c in range(self.reduced_components):
                klt = self.T_temp[c]
                S[:, :, c] = klt.transform(S[:, :, c]) 
            S = self.model.agg(S, agg)
        else:
            S = self.model.transform(X, exponent=exponent, agg=agg, act=act, resample=resample)
        return S

    def hotelling(self, X):
        """
        https://math.unm.edu/~james/w5-STAT576b.pdf
        """
        n = self.model.n_tiles
        y_true = self.model.net.whiten.mean_.data.cpu().numpy()
        y_means = self.model.i2col(torch.FloatTensor(X)).view(n, len(X), -1).mean(0).cpu().numpy() - y_true
        conv_inv = np.linalg.pinv(self.model.cov).astype(np.float32)
        Z = n*np.einsum("ij,ji->i", y_means, conv_inv @ y_means.T) # inner product
        return Z

    def typical_set(self, X, bs=4096, elbo=False):
        mean = self.model.net.whiten.mean_.data.cpu().numpy()
        
        N, T, F = len(X), self.model.n_tiles, len(mean)
        centered_patches = self.model.i2col(torch.FloatTensor(X)).view(T, N, F) - mean.reshape(1, 1, F) 
        centered_patches = centered_patches.reshape(T*N, F)
        log_probs = []
        for i in np.arange(0, T*N, bs):
            if elbo:
                log_probs.append(super(type(self.model), self.model).elbo(centered_patches[i:i+bs], p_z=Loss.Gaussian).unsqueeze(1))
            else:
                log_probs.append(super(type(self.model), self.model).log_prob(centered_patches[i:i+bs]).unsqueeze(1))
        log_probs = torch.cat(log_probs, dim=0).reshape(T, N)
        return log_probs.mean(0).detach().numpy()

    def mean_shift(self, X):
        mean = self.model.net.whiten.mean_.data.cpu().numpy()
        N, T, F = len(X), self.n_tiles, len(mean)
        centered_patches = self.model.i2col(torch.FloatTensor(X)).view(T, N, F) - mean.reshape(1, 1, F) 
        return centered_patches.mean(0).detach().numpy()

    def avg_patch_reconstruct(self, X):
        patches = self.model.i2col(torch.FloatTensor(X))
        patch_reconstructions = self.model.predict(patches)[0]
        mse = torch.norm(patches - patch_reconstructions , dim=1).numpy()
        mse = mse[im2colOrder(len(X), len(patches))]
        avg_mse = mse.reshape(len(X), self.model.n_tiles).mean(1)
        return avg_mse

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