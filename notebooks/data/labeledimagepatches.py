# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import warnings
from torchvision import transforms
from imageio import imread
import os
from skimage.transform import warp, AffineTransform
from imagepatches import ImagePatches

__all__ = ["LabeledImagePatches"]


class LabeledImagePatches(ImagePatches):
    """
    Dataset for generating data from a single given image with labeled defects in a separate mask file. 
    It used a window-scheme, hence the name Image Patches.
    
    Arguments:
        * file (str) : The image filename
        * file (str) : The mask image filename
        * mode (str) : The processing mode 'bgr' or 'gray' or 'rgb' or 'gray3channel' (default="bgr")
        * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
        * crop (list) : tlx, tly, brx, bry, default [0, 0, -1, -1]
        * padding (list) : tlx, tly, brx, bry, default [0, 0, 0, 0] gets applied after crop
        * limit (int) : limits the number of patches (default: -1)
        * shuffle (boolean) : random pick limited patches (default: False)
        * oneclass (bool): only return good samples as training examples
        * affine_map (ndarray) : 3x3 ndarray defining the affine transformation (must be specified as inverse, because skimage is performing reverse mapping)
        * anomaly_offset_percentage (int) : Offset of an anomaly to the center of a patch in percent to the width/height of the patch (default: 10)
    """
    
    def __init__(self, file, mask_file=None, train_label=0, test_label=1, oneclass=False, anomaly_offset_percentage=10, *args, **kwargs):
        super().__init__(file, *args, **kwargs)
        self.train_label = train_label 
        self.test_label = test_label 
        self.oneclass = oneclass
        self.anomaly_offset_percentage = anomaly_offset_percentage

        assert self.anomaly_offset_percentage > 0

        # Mask image >> only keep the first layer
        self.mask_image = None
        if mask_file is not None:
            if type(mask_file) is not list: mask_file = [mask_file]
            self.mask_image = imread(mask_file[0])
            assert self.mask_image.ndim == 2 or (self.mask_image.ndim == 3 and self.mask_image.shape[2] == 1)
            for f in mask_file[1:]:
                try:
                    self.mask_image = np.vstack((self.mask_image, imread(f))) # (20, 20, 3), (20, 20, 3) --> (40, 20, 3)
                except:
                    self.mask_image = np.hstack((self.mask_image, imread(f))) # (20, 20, 3), (20, 20, 3) --> (20, 40, 3)
            if self.mask_image.ndim == 2: 
                self.mask_image = np.expand_dims(np.asarray(self.mask_image), 2)
            # Cropping
            self.mask_image = self.crop(self.mask_image, (self.tlx, self.tly, self.brx, self.bry))
            # Padding
            self.mask_image = self.padd(self.mask_image, (self.ptlx, self.ptly, self.pbrx, self.pbry))
        else:
            self.mask_image = np.zeros(self.img.shape[0:2]) 
            if self.mask_image.ndim == 2: 
                self.mask_image = np.expand_dims(self.mask_image, 2)

        assert self.img.shape[:2] == self.mask_image.shape[:2]
        assert self.mask_image.ndim == 3

        # Apply given affine transformation to mask_image
        if self.affine_transform is not None:
            self.mask_image = warp(self.mask_image, self.affine_transform, preserve_range=True).astype(np.uint8)

        # compute labels
        self.idx_mapping, self.all_labels, self.num_train_samples, self.num_test_samples = self._label_image()

    def __len__(self):
        if self.train:
            return self.num_train_samples
        else:
            return self.num_test_samples  

    def _label_image(self):
        good_idx, defective_idx = self._compute_labeled_patches(self.limit, self.shuffle)
        idx_mapping = good_idx + defective_idx
        all_labels = np.zeros(self.dataset_size).astype(np.int32)
        all_labels[good_idx] = self.train_label
        all_labels[defective_idx] = self.test_label              
        # case one class setting
        if self.oneclass == True:
            if self.train == True:
                idx_mapping = good_idx
            else:
                idx_mapping = defective_idx
                
            # add unlearnt training samples to test set
            if self.train_percentage < 1.0:
                if self.train == True:
                    num_train_samples = int(self.train_percentage * len(good_idx))
                    idx_mapping =  good_idx[:num_train_samples]
                else:
                    num_train_samples = int(self.train_percentage * len(good_idx))
                    unlearnt_training_samples = good_idx[num_train_samples:]
                    idx_mapping =  defective_idx + unlearnt_training_samples

            # store dataset sizes            
            split = int(len(good_idx) * self.train_percentage)   
            num_train_samples = split
            num_test_samples = len(defective_idx) + len(good_idx) - split
        else:
            # store dataset sizes
            split = int(len(idx_mapping) * self.train_percentage)   
            num_train_samples = split
            num_test_samples = len(idx_mapping) - split
        return idx_mapping, all_labels, num_train_samples, num_test_samples

    def _compute_labeled_patches(self, limit=-1, shuffle=False):
        # Handle limits in case we do not want to process the whole image
        if limit == -1:
            limit = self.dataset_size
        idx = np.arange(self.dataset_size)
        # Randomize the access indices so that we process arbitrary positions
        if shuffle:
            idx = np.random.permutation(idx)
        good_idx, defective_idx, filtered_idx = [], [], []
        # Loop through randomized indices until limit reached and fill label buckets
        for i in idx:
            mask = self._get_internal_patch(i, self.mask_image)
        
            if np.max(mask) > 0:
                # test if the error is too close to the border
                width = mask.shape[1]
                height = mask.shape[0]
                center_x = width // 2
                center_y = height // 2
                y, x = np.meshgrid(np.arange(center_y - int(center_y / 100 * self.anomaly_offset_percentage),
                                            center_y + int(center_y / 100 * self.anomaly_offset_percentage)),
                                np.arange(center_x - int(center_x / 100 * self.anomaly_offset_percentage),
                                            center_x + int(center_x / 100 * self.anomaly_offset_percentage)))
                if np.max(mask[y, x]) > 0:
                    defective_idx.append(i)
                else:
                    filtered_idx.append(i)
                    if len(filtered_idx) < 2:
                        print("Anomaly is not in the center of the patch ({})".format(int(center_y / 100 * self.anomaly_offset_percentage)))
                #if (np.sum(mask) / np.prod(mask.shape)) < 0.05:
                #     print("Number of overlapping Pixels of Mask and Patch is very small < 1%")
                # TODO: use area percentage as decision criterion ´labeledPx/allPxInPatch > 0.3´

            else:
                good_idx.append(i)
            if len(good_idx) == limit and len(defective_idx) >= limit:
                break
            if len(defective_idx) == limit and len(good_idx) >= limit:
                break
        if len(filtered_idx) > 1:
            print(f"{len(filtered_idx)} patches skipped because error was too close to the border.")
        if limit > 0:
            return good_idx[:limit], defective_idx[:limit]
        else:
            return good_idx, defective_idx
    
    
    def _get_label(self, idx):
         idx = self.idx_mapping[idx]
         return int(self.all_labels[idx])

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()

        patch = self.get_patch(idx)
        label = self._get_label(idx)
        if self.transform:
            patch = self.transform(patch)
        return patch, label
