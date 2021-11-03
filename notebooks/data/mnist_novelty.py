import os
import os.path as osp
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import StandardScaler
import numpy as np

class MNIST_OneClass():

    '''
    A small MNIST wrapper to train one-class classifier a.k.a. novelty detection

    X, X_in, X_out = MNIST_OneClass("./data", train_classes=[0], test_classes=[1, 2], inlier_partition=0.95, transform=transforms.ToTensor())

    '''

    def __init__(self,
                 root="./data",
                 train_classes=[0],
                 test_classes=[1],
                 transform=transforms.ToTensor(),
                 download=True,
                 z_normalize=False,
                 balanced_test_set=True):

        raw_train = MNIST(root=root, download=download, train=True, transform=transform)
        raw_test = MNIST(root=root, download=download, train=False, transform=transform)

        data_train = [raw_train[i][0] for i in range(len(raw_train)) if raw_train[i][1] in train_classes]
        data_train = torch.cat(data_train).reshape(len(data_train), -1).numpy()
        
        data_test_outliers = [raw_test[i][0] for i in range(len(raw_test)) if raw_test[i][1] in test_classes]
        data_test_outliers = torch.cat(data_test_outliers).reshape(len(data_test_outliers), -1).numpy()

        data_test_inliers = [raw_test[i][0] for i in range(len(raw_test)) if raw_test[i][1] in train_classes]
        data_test_inliers = torch.cat(data_test_inliers).reshape(len(data_test_inliers), -1).numpy()

        # shuffle train data
        permut = np.random.permutation(np.arange(len(data_train)))
        data_train = data_train[permut]
        
        # shuffle test data
        permut = np.random.permutation(np.arange(len(data_test_outliers)))
        data_test_outliers = data_test_outliers[permut]

        # shuffle test data
        permut = np.random.permutation(np.arange(len(data_test_inliers)))
        data_test_inliers = data_test_inliers[permut]

        # balance test data
        if balanced_test_set:
            mini = min([data_test_inliers.shape[0], data_test_outliers.shape[0]])
            data_test_inliers = data_test_inliers[:mini]
            data_test_outliers = data_test_outliers[:mini]
        
        # members 
        self.z_normalize = z_normalize
        self.train_classes = train_classes
        self.test_classes = test_classes

        # Provide Design Matrices as members
        if self.z_normalize:
            print("Standardizing data (with_mean=True, with_std=True)")
            scaler = StandardScaler(with_mean=True, with_std=True).fit(data_train)
            self.data_train = scaler.transform(data_train)
            self.data_test_inliers  = scaler.transform(data_test_inliers)
            self.data_test_outliers = scaler.transform(data_test_outliers)
        else:
            self.data_train = data_train
            self.data_test_inliers  = data_test_inliers
            self.data_test_outliers = data_test_outliers            

        print(self)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data_train, self.data_test_inliers, self.data_test_outliers

    def __repr__(self): 
        return "{}(z_normalize={}, train_classes={}, test_classes={}, data_train={}, data_test_inliers={}, data_test_outliers={})".format(
            self.__class__.__name__,
            self.z_normalize,
            self.train_classes, self.test_classes, self.data_train.shape, self.data_test_inliers.shape, self.data_test_outliers.shape)

if __name__ == "__main__":
    data_train, data_test_inliers, data_test_outliers = MNIST_OneClass()[0]