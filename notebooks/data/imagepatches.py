# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
from torchvision import transforms
from imageio import imread
import numpy as np
from skimage.transform import warp, AffineTransform


class ImagePatches(Dataset):
    """
    Dataset for generating data from a single given image. It used a window-scheme, hence the name ImageTiles.
    
    Arguments:
        * file (str) : The image filename or list of files
        * mode (str) : The processing mode 'bgr' or 'gray' or 'rgb' or 'hsv' or 'gray3channel' or 'height' or 'height3channel' (default="bgr")
        * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
        * crop (list) : tlx, tly, brx, bry, default [0, 0, -1, -1]
        * padding (list) : tlx, tly, brx, bry, default [0, 0, 0, 0] is applied after cropping
        * limit (int) : limits the number of patches (default: -1)
        * shuffle (boolean) : random pick limited patches (default: False)
        * affine_map (ndarray) : 3x3 ndarray defining the affine transformation (must be specified as inverse, because skimage is performing reverse mapping)
    """
    
    def __init__(self, file, mode='bgr', mean_normalization=False, train = True, train_percentage=.8, transform=transforms.Compose([transforms.ToTensor()]),
                 stride_x=16, stride_y=16, window_size=32, label=1, crop=[0, 0, -1, -1], padding=[0, 0, 0, 0], limit=-1, shuffle=False, affine_map=None):

        self.filename = file
        if type(file) is not list: file = [file]
        self.img = imread(file[0])
        for f in file[1:]:
            try:
                # try to stack vertivally
                self.img = np.vstack((self.img, imread(f) )) # (20, 20, 3), (20, 20, 3) --> (40, 20, 3)
            except:
                try:
                    # try to stack horizontally
                    self.img = np.hstack((self.img, imread(f) )) # (20, 20, 3), (20, 20, 3) --> (20, 40, 3)
                except:
                    # unfortunately we have to crop to fit the smaller image
                    img_new = imread(f)
                    h_diff = abs(self.img.shape[0] - img_new.shape[0])
                    v_diff = abs(self.img.shape[1] - img_new.shape[1])
                    if h_diff < v_diff:
                        # stack vertically
                        w = min(self.img.shape[1], img_new.shape[1])
                        print("Input images have different shapes. Cropping original image from shape {} to new shape {}".format(self.img.shape, self.img[:,:w,:].shape))
                        self.img = np.vstack((self.img[:,:w,:], img_new[:,:w,:])) # (20, 20, 3), (20, 20, 3) --> (40, 20, 3)
                    else:
                        # stack horizontally
                        h = min(self.img.shape[0], img_new.shape[0])
                        print("Input images have different shapes. Cropping original image from shape {} to new shape {}".format(self.img.shape, self.img[:h,:,:].shape))
                        self.img = np.hstack((self.img[:h,:,:], img_new[:h,:,:])) # (20, 20, 3), (20, 20, 3) --> (20, 40, 3)

        self.train = train
        self.train_percentage = train_percentage
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.window_size = window_size
        self.transform = transform
        self.label = label
        self.limit = limit
        self.shuffle = shuffle

        self.tlx, self.tly, self.brx, self.bry = crop
        self.ptlx, self.ptly, self.pbrx, self.pbry = padding

        # Normalize patch to (H, W, C) with [0, 1] float32
        assert self.img.min() >= 0
        self.img = self.img.astype(np.float32)
        while self.img.max() > 1. : # while as some images are scaled larger than 255 (e.g. disparity images)
            self.img = self.img / 255
        if self.img.ndim == 3 and self.img.shape[0] < self.img.shape[2] and self.img.shape[0] < self.img.shape[1]:
            # bring channel to back
            self.img = np.transpose(self.img, (1,2,0))
        assert self.img.min() >= 0 and self.img.max() <= 1
        
        # handle gray scale
        if self.img.ndim == 2:
            self.img = np.expand_dims(self.img,2)
        assert self.img.ndim == 3

        # handle transparent channel in png
        if self.img.shape[2] == 4:
            warnings.warn("I dont know how to handle transparency and will skip 4th channel (img[:,:,3]). Shape of image:" + str(self.img.shape), RuntimeWarning)       
            self.img = self.img[:,:,0:3]

        # handle RGB and store a copy
        if mode == 'bgr':
            self.img = self._rgb_to_bgr()
        elif mode == 'gray':
            self.img = self._rgb_to_gray()
        elif mode == 'gray3channel':
            self.img = self._rgb_to_gray3channel()
        elif mode == 'height':
            self.img = self._to_height()
        elif mode == 'height3channel':
            self.img = self._to_height3channel()
        elif mode == 'hsv':
            self.img = self._rgb_to_hsv()

        if mean_normalization:
            if mode == 'gray':
                self.img[:, :, 0] = self.img[:, :, 0] - np.mean(self.img[:, :, 0])
            else:
                self.img[:, :, 0] = self.img[:, :, 0] - np.mean(self.img[:, :, 0])
                self.img[:, :, 1] = self.img[:, :, 1] - np.mean(self.img[:, :, 1])
                self.img[:, :, 2] = self.img[:, :, 2] - np.mean(self.img[:, :, 2])

            #rescale 01
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        if self.brx == -1:
            self.brx = self.img.shape[1]
        if self.bry == -1:
            self.bry = self.img.shape[0]

        # Cropping
        self.img = self.crop(self.img, (self.tlx, self.tly, self.brx, self.bry))
        
        # Padding
        self.img = self.padd(self.img, (self.ptlx, self.ptly, self.pbrx, self.pbry))

        # Apply given affine transformation to image and mask_image
        self.affine_transform = None
        if affine_map is not None:
            self.affine_transform = AffineTransform(affine_map)
            self.img = warp(self.img, self.affine_transform, preserve_range=True).astype(np.float32)

        self.patches_per_y = (((self.img.shape[0] - self.window_size) // self.stride_y) + 1)
        self.patches_per_x = (((self.img.shape[1] - self.window_size) // self.stride_x) + 1)
        self.dataset_size = int(self.patches_per_y) * int(self.patches_per_x)
        
        # Shuffle the dataset
        self.idx_mapping = np.arange(self.dataset_size) 
        if self.shuffle:
            self.idx_mapping  = np.random.permutation(self.idx_mapping)
        
        # Enforce limits
        if self.limit > -1 and self.dataset_size > self.limit:
            self.dataset_size = limit

    def crop(self, im, crop):
        tlx, tly, brx, bry = crop
        if im.ndim == 3:
            im      = im[tly : bry, tlx:brx , :].copy()
        elif im.ndim == 2:
            im      = im[tly: bry, tlx:brx].copy()
        else:
            raise AttributeError(im.shape + ' image dimensions invalid.')
        return im
    
    def padd(self, im, padding):
        ptlx, ptly, pbrx, pbry = padding
        if im.ndim == 3:
            image = np.zeros((im.shape[0] + ptly + pbry, im.shape[1] + ptlx + pbrx, im.shape[2])).astype(im.dtype)
            image[ptly : ptly + im.shape[0], ptlx : ptlx + im.shape[1] , :] = im.copy()
            im      = image
        elif im.ndim == 2:
            image = np.zeros((im.shape[0] + ptly + pbry, im.shape[1] + ptlx + pbrx))
            image[ptly : ptly + im.shape[0], ptlx : ptlx + im.shape[1]] = im.copy()
            im      = image
        else:
            raise AttributeError(im.shape + ' image dimensions invalid.')
        return im
    
    def _rgb_to_bgr(self):
        assert self.img.shape[2] == 3
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        return np.concatenate((b,g,r), axis=2).copy()
    
    def _rgb_to_hsv(self):
        assert self.img.shape[2] == 3
        from matplotlib.colors import rgb_to_hsv
        img = rgb_to_hsv(self.img)
        return img.copy()

    def _rgb_to_gray(self):
        if self.img.shape[2] == 1:
            return self.img
        assert self.img.shape[2] == 3
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        return ((.2989 * r) + (.5870 * g) + (.114 * b)).copy()

    def _rgb_to_gray3channel(self):
        img = np.squeeze(self.img)
        if img.ndim == 2:
            g = np.expand_dims(img, axis=2)
            return  np.concatenate((g, g, g), axis=2)   
        if self.img.ndim == 3:
            r = np.expand_dims(self.img[:, :, 0], axis=2)
            g = np.expand_dims(self.img[:, :, 1], axis=2)
            b = np.expand_dims(self.img[:, :, 2], axis=2)
            img = (.2989 * r) + (.5870 * g) + (.114 * b)
            return np.concatenate((img, img, img), axis=2).copy()

    def _to_height(self):
        assert self.img.ndim == 2 or self.img.shape[2] == 1
        assert self.img.min() >= 0.
        img = self.img - self.img.min() # [0, max]
        img = img / img.max() # [0, 1]
        return img

    def _to_height3channel(self):
        assert self.img.ndim == 2 or self.img.shape[2] == 1
        assert self.img.min() >= 0.
        img = self.img - self.img.min() # [0, max]
        img = img / img.max() # [0, 1]
        img = np.concatenate((img, img, img), axis=2).copy()
        return img

    def stats(self):
        return {
            "name"  : "ImagePatches",
            "filepath" : self.filename,
            "data split" : self.train_percentage,
            "data set" : "train" if self.train else "test",
            "data samples": len(self),
            "data shape" : self.__getitem__(0)[0].shape,
            "data dtype" : self.__getitem__(0)[0].dtype,
            "data label example" : self.__getitem__(0)[1]
            }
    
    def __repr__(self):
        return str(self.stats())
        
    def __len__(self):
        if self.train:
            return int(self.train_percentage * self.dataset_size)
        else:
            return int((1 - self.train_percentage) * self.dataset_size)

    def _draw_boarder(self, patch, color, thickness, inplace = True):
         if thickness == 0: return patch
         if inplace == False: patch = patch.copy()
         if patch.shape[2] < 3:
             patch = np.concatenate((patch, patch, patch), axis=2).copy()
         if np.max(patch) > 1 and np.sum(color) <= 3:
             color = np.asarray(color) * 255.
         if np.max(patch) <= 1 and np.sum(color) > 3:
             color =  np.asarray(color) / 255.
         patch[0:thickness,:, :] = color # top
         patch[-thickness:,:, :] = color # bottom
         patch[:,0:thickness, :] = color # left
         patch[:,-thickness:, :] = color # left
         return patch

    def mark_area(self, patch_indices, color=(255,255,255), thickness=2, image=None, union=False):
        """
        Same as mark_patches, but with non-maximum suppression.
        """
        # INPUT VALIDATION
        # ensure patch_indices is ndarray
        if isinstance(patch_indices, list) == False and isinstance(patch_indices, np.ndarray) == False:
            patch_indices = [patch_indices]
        patch_indices = np.asarray(patch_indices).squeeze() if len(patch_indices) > 1 else np.asarray(patch_indices)
        if patch_indices.ndim == 2:
            patch_indices = patch_indices[:,0] #in case of multiple columns take the first
        # handle target image
        if image is None:
            # Compute on a copy
            marked_img = self.img.copy()
        else:
            marked_img = image.astype(np.float32)
        # scale between 0 and 1
        if marked_img.max() > 1:
            marked_img = marked_img / 255.
        if marked_img.ndim == 2:
            marked_img = np.expand_dims(marked_img, 2)
        assert self.img.shape[:2] == marked_img.shape[:2]

        # transform indices to internal indecies
        patch_indices = [self.idx_mapping[i] for i in patch_indices]
        boxes = np.zeros((len(patch_indices), 4), dtype=np.int32)
        for bb, i in enumerate(patch_indices):
            # get the bounding boxes
            boxes[bb, :] = self._get_bounding_box(i)

        # non maximum suppression
        if union:
            boxes = nms.unify(boxes)
        else:
            boxes = nms.non_max_suppression_fast(boxes, overlapThresh=0)

        # draw boxes
        for i, box in enumerate(boxes):
            topleft_x, topleft_y, bottomright_x, bottomright_y = box
            area = marked_img[topleft_y:bottomright_y, topleft_x:bottomright_x,:]
            self._draw_boarder(area, color, thickness, inplace=True)

        #gray scale
        marked_img = marked_img.squeeze()
        return (marked_img * 255).astype(np.uint8)

    def mark_patches(self, patch_indices, color=(255, 255, 255), thickness=2, image = None):
        if isinstance(patch_indices, list) == False and isinstance(patch_indices, np.ndarray) == False:
            patch_indices = [patch_indices]
        patch_indices = np.asarray(patch_indices).squeeze() if len(patch_indices) > 1 else np.asarray(patch_indices)
        if patch_indices.ndim == 2:
            patch_indices = patch_indices[:,0] #in case of multiple columns take the first

        if isinstance(color, tuple):
            color = [color] * len(patch_indices)
        if image is None:
            # Compute on a copy
            marked_img = self.img.copy()
        else:
            marked_img = image.astype(np.float32)
            if marked_img.max() > 1:
                marked_img = marked_img / 255.
            if marked_img.ndim == 2:
                marked_img = np.expand_dims(marked_img, 2)
            assert self.img.shape[:2] == marked_img.shape[:2]

        for i, c in zip(patch_indices, color):
            patch = self.get_patch(i, marked_img)
            # Draw line..
            patch = self._draw_boarder(patch, c, thickness, inplace=True)
            # Copy to image
            marked_img = self._put_patch(marked_img, i, patch)
        #gray scale
        marked_img = marked_img.squeeze()
        return (marked_img * 255).astype(np.uint8)

    def _get_bounding_box(self, internal_idx):
        x = internal_idx % self.patches_per_x
        y = internal_idx // self.patches_per_x

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size

        return topleft_x, topleft_y, bottomright_x, bottomright_y

    def _get_internal_patch(self, internal_idx, img):
        topleft_x, topleft_y, bottomright_x, bottomright_y = self._get_bounding_box(internal_idx)
        patch = img[topleft_y : bottomright_y, topleft_x : bottomright_x, :].copy()
        
        return patch
    
    def get_patch(self, idx, img = None):
        idx = self.idx_mapping[idx]
        if img is None:
            patch = self._get_internal_patch(idx, self.img)
        else:
            patch = self._get_internal_patch(idx, img)

        return patch

    def _put_patch(self, img, idx, patch):
        # map external index to internal
        idx = self.idx_mapping[idx]
        x = idx % self.patches_per_x
        y = idx // self.patches_per_x

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size

        if patch.ndim == 2: np.expand_dims(patch, 2)
        if patch.shape[2] > img.shape[2]:
             img = np.concatenate((img, img, img), axis=2).copy()
        img[topleft_y : bottomright_y, topleft_x : bottomright_x, :] = patch
        return img

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
            
        patch = self.get_patch(idx)
        label = self.label
        if self.transform:
            patch = self.transform(patch)
        return patch, label
