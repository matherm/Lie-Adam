import torch
from torchvision.transforms import transforms
import glob
import numpy as np
from imageio import imread
import re
from PIL import Image
from skimage.transform import resize
from collections import namedtuple
from hugeica.helpers import im2col, im2colOrder

from albumentations import (
    ImageOnlyTransform,
    BasicTransform,
    Compose,
    OneOf,
    MotionBlur,
    MedianBlur,
    Blur,
    CLAHE,
    Sharpen,
    Emboss,
    RandomBrightnessContrast,
    HueSaturationValue,
    GaussNoise,
    HorizontalFlip,
    RandomRotate90,
    ShiftScaleRotate,
)

def RGB(x):
    return np.asarray( x.convert("RGB") )

def NOP(image):
    return image

class MVTEC(torch.utils.data.Dataset):
    
    CLASSES = ["bottle", "carpet" , "leather", \
               "pill", "tile", "wood", "cable", \
               "grid", "toothbrush" , \
               "zipper", "capsule", "hazelnut", \
               "metal_nut", "screw", "transistor"]
    
    def __init__(self, dataset="train", path="../../../../nas-files/mvtec", clazz="bottle", augment=False, load_size=256, crop_size=224, normalize=True):
        
        if dataset=="train":
            self.paths = glob.glob(path + "/" + clazz + "/train/good/**.png")
        if dataset=="outlier":    
            self.paths = [fn for fn in glob.glob(path + "/" + clazz + "/test/**/**.png") if not "good" in fn]
            self.labels = glob.glob(path + "/" + clazz + "/ground_truth/**/**.png")
        if dataset=="inlier":
            self.paths  = glob.glob(path + "/" + clazz + "/test/good/**.png")
        
        self.size = crop_size
        #self.to_image = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256,256)), transforms.CenterCrop((224, 224)), RGB])
        
        
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        
        
        if normalize:
            self.data_transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((load_size, load_size)),#, Image.ANTIALIAS),
                            transforms.ToTensor(),
                            transforms.CenterCrop(crop_size),
                            transforms.Normalize(mean=mean_train, std=std_train)
                            ])
        else:
            self.data_transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((load_size, load_size)),#, Image.ANTIALIAS),
                            transforms.ToTensor(),
                            transforms.CenterCrop(crop_size),
           #                 transforms.Normalize(mean=mean_train, std=std_train)
                            ])
            
        self.gt_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((load_size, load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(crop_size)
                        ])
        
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
             
        if augment:
            info = MVTEC.augmentation_info(clazz)
            self.augment = detection_aug(p=0.5, **info)
        else:
            self.augment = NOP
            

            
        
    def label_for_image(self, idx):
        
        path = self.paths[idx]
        
        fname = re.match(".*/(\d\d\d).png" , path)[1] + "_mask.png"  # image number
        aname = re.match(".*/(.*)/.*.png" , path)[1]                 # anomaly name, e.g. 'hole'
        
        label = torch.zeros((1, self.size, self.size))
        
        for lpath in self.labels:
            if fname in lpath and aname in lpath:  
                mask = self.gt_transforms(imread(lpath))
                label += mask[:, :]
        
        label[label > 0] = 1
        return label
            
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert('RGB')
        im = np.asarray(im)
        #im = imread()
        #im = self.to_image(im)
        im = self.augment(image=im)
        if type(im) is not np.ndarray:
            im = im["image"]
        im =  self.data_transforms(im)
        return im
    
    
    @staticmethod
    def augmentation_info(category):
        """Augmentation used for anomaly detection tasks on an MVTec category."""
        return {
            # Some MVTec categories contain text or directions so they must not be flipped
            "flip": (
                category not in ["metal_nut", "pill", "capsule", "screw"]
            ),
            # Some categories may not be rotated (because a 90° rotation is an anomaly)
            "rotate90": (category not in ["transistor"]),
            # Some smaller rotations produce edge artifacts for some categories
            "rotate45": (
                category
                not in [
                    "bottle",
                    "cable",
                    "capsule",
                    "metal_nut",
                    "pill",
                    "screw",
                    "toothbrush",
                    "transistor",
                    "wood",
                    "zipper",
                ]
            ),
            # Some patterns are really messed up with shifts/scales on the edges
            "background_edge": (category not in ["grid"]),
            "noise": False,
        }

    @staticmethod
    def anomaly_info(hparam):
        """Meta-information on anomalies for training with this dataset.
        Returned dict should contain:
            "min_size": minimal size of an anomaly, as fraction of the image
                area.
            "size_quantiles": q = 5 quantiles of anomaly sizes, given as q + 1
                boundary values in increasing order.
        """
        # Generated from src.scripts.min_anomaly_size.
        min_anomaly_sizes = {
            "bottle": 0.00575679012345679,
            "cable": 0.0014362335205078125,
            "capsule": 0.000371,
            "carpet": 0.001499176025390625,
            "grid": 0.0007152557373046875,
            "hazelnut": 0.0023288726806640625,
            "leather": 0.000858306884765625,
            "metal_nut": 0.0011959183673469387,
            "pill": 0.0003171875,
            "screw": 0.000659942626953125,
            "tile": 0.008421201814058957,
            "toothbrush": 0.00141143798828125,
            "transistor": 0.0022373199462890625,
            "wood": 0.001071929931640625,
            "zipper": 0.0015916824340820312,
        }
        # Generated from code.scripts.anomaly_size_distribution.
        return {"min_size": min_anomaly_sizes[hparams.category]}
    
def detection_aug(
    p: float = 0.5,
    flip: bool = True,
    rotate90: bool = True,
    noise: bool = True,
    rotate45: bool = True,
    background_edge: bool = True,
) -> Compose:
    """Augmentation used for anomaly detection tasks."""
    augs = [
        pixel_aug(p=1, noise=noise),
    ]
    if flip:
        # Only horizontal flips as vertical is same as horizontal + 180°
        # rotate.
        augs.append(HorizontalFlip())
    if rotate90:
        augs.append(RandomRotate90())
        augs.append(
            ShiftScaleRotate(
                shift_limit=0.05 if background_edge else 0,
                #shift_limit=0.2 if background_edge else 0,
                scale_limit=(-0.05, 0.1 if background_edge else 0),
                rotate_limit=(45 if rotate45 else 15) if background_edge else 0,
                p=0.2,
            )
    )
    return Compose(augs, p=p)
    
def pixel_aug(p: float = 0.5, noise: bool = True) -> Compose:
    """Augmentation only on a pixel-level."""
    augs = [
        OneOf(
            [
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        OneOf(
            [
                CLAHE(clip_limit=2),
                Sharpen(),
                Emboss(),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.2)),
            ],
            p=0.3,
        ),
        # Reduced hue shift to not change the color that much (purple
        # hazelnuts).
        # reduced val shift to not overly darken the image
        HueSaturationValue(
            hue_shift_limit=10, val_shift_limit=(-10, 20), p=0.3
        ),
    ]
    if noise:
        augs.append(
            OneOf(
                [
                    # Slightly less aggressive:
                    GaussNoise(
                        std=(0.01 * 255, 0.03 * 255), per_channel=False
                    ),
                    GaussNoise(
                        std=(0.01 * 255, 0.03 * 255), per_channel=True
                    ),
                ],
                p=0.2,
            )
        )
    return Compose(augs, p=p)

def dataset_to_patches(X_, P, s):
    N, C = len(X_), X_.shape[1]
    X = im2col(X_,  BSZ=(P, P), padding=0, stride=s).T
    return X[im2colOrder(N, len(X))].reshape(len(X), C, P, P)
    

def dataloader(clazz, P=224, s=112, label_per_patch=False, MVTEC_PATH="../../../../nas-files/mvtec", augment=False, load_size=256, crop_size=224, normalize=True):
    X = MVTEC(dataset="train", path=MVTEC_PATH, clazz=MVTEC.CLASSES[clazz], augment=augment, load_size=load_size, crop_size=crop_size, normalize=normalize)
    X_ = np.stack([im.numpy() for im in X])

    X_valid_ = MVTEC(dataset="inlier", path=MVTEC_PATH, clazz=MVTEC.CLASSES[clazz], augment=augment, load_size=load_size, crop_size=crop_size, normalize=normalize)
    X_valid_ = np.stack([im.numpy() for im in X_valid_])

    X_test = MVTEC(dataset="outlier", path=MVTEC_PATH, clazz=MVTEC.CLASSES[clazz], augment=augment, load_size=load_size, crop_size=crop_size, normalize=normalize)
    X_test_ = np.stack([im.numpy() for im in X_test])
    X_labels = np.stack([X_test.label_for_image(i) for i in range(len(X_test))])

    X_ = dataset_to_patches(X_, P, s)
    X_valid_ = dataset_to_patches(X_valid_, P, s)
    X_test_ = dataset_to_patches(X_test_, P, s)
    X_labels = dataset_to_patches(X_labels, P, s)

    if label_per_patch:
        X_test_no_defect = X_test_[X_labels.reshape(len(X_labels), -1).max(1) == 0.]
        X_test_defects = X_test_[X_labels.reshape(len(X_labels), -1).max(1) == 1]
        X_labels_ = X_labels[X_labels.reshape(len(X_labels), -1).max(1) == 1]

        assert len(X_test_no_defect) + len(X_test_defects) == len(X_labels)
        assert len(X_test_no_defect) + len(X_test_defects) == len(X_test_)

        X_valid_ = np.concatenate([X_valid_, X_test_no_defect])
        X_test_ = X_test_defects
        X_labels = X_labels_

    if X_[0].shape[0] == 1:
        X_ = np.repeat(X_, 3, axis=1)
        X_valid_ = np.repeat(X_valid_, 3, axis=1)
        X_test_ = np.repeat(X_test_, 3, axis=1)
    
    return X_, X_valid_, X_test_, X_labels, len(X_)//len(X)