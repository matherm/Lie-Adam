import glob, imageio
from imagepatches import *
from labeledimagepatches import *
from iosdata.utils.nasfile import NasFile
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler


FILES = [
        (
        "./data/scans_labeled/IMG_4/original_size/without_error/SKopierer619051610220_0039.bmp",
        "./data/scans_labeled/IMG_4/original_size/without_error/SKopierer619051610220_0040.bmp",
        "./data/scans_labeled/IMG_4/original_size/with_error_2/SKopierer619051610220_0020.bmp",
        "./data/scans_labeled/IMG_4/original_size/error_mask_2/SKopierer619051610220_0020.png"),
        (
        "./data/scans_labeled/IMG_4/original_size/without_error/SKopierer619051610220_0039.bmp",
        "./data/scans_labeled/IMG_4/original_size/without_error/SKopierer619051610220_0040.bmp",
        "./data/scans_labeled/IMG_4/original_size/with_error_1/SKopierer619051610220_0019.bmp",
        "./data/scans_labeled/IMG_4/original_size/error_mask_1/SKopierer619051610220_0019.png"),
        (
        "./data/scans_labeled/IMG_22/original_size/without_error/SKopierer619051610220_0026.bmp",
        "./data/scans_labeled/IMG_22/original_size/without_error/SKopierer619051610220_0025.bmp",
        "./data/scans_labeled/IMG_22/original_size/with_error_2/SKopierer619051610220_0006.bmp",
        "./data/scans_labeled/IMG_22/original_size/error_mask_2/SKopierer619051610220_0006.png"),
        (
        "./data/scans_labeled/IMG_22/original_size/without_error/SKopierer619051610220_0026.bmp",
        "./data/scans_labeled/IMG_22/original_size/without_error/SKopierer619051610220_0025.bmp",
        "./data/scans_labeled/IMG_22/original_size/with_error_1/SKopierer619051610220_0005.bmp",
        "./data/scans_labeled/IMG_22/original_size/error_mask_1/SKopierer619051610220_0005.png"),
        (
        "./data/scans_labeled/IMG_6/original_size/without_error/SKopierer619051610220_0037.bmp",
        "./data/scans_labeled/IMG_6/original_size/without_error/SKopierer619051610220_0038.bmp",
        "./data/scans_labeled/IMG_6/original_size/with_error_2/SKopierer619051610220_0018.bmp",
        "./data/scans_labeled/IMG_6/original_size/error_mask_2/SKopierer619051610220_0018.png"),
        (
        "./data/scans_labeled/IMG_6/original_size/without_error/SKopierer619051610220_0037.bmp",
        "./data/scans_labeled/IMG_6/original_size/without_error/SKopierer619051610220_0038.bmp",
        "./data/scans_labeled/IMG_6/original_size/with_error_1/SKopierer619051610220_0017.bmp",
        "./data/scans_labeled/IMG_6/original_size/error_mask_1/SKopierer619051610220_0017.png"),
        (
        "./data/scans_labeled/IMG_8/original_size/without_error/SKopierer619051610220_0035.bmp",
        "./data/scans_labeled/IMG_8/original_size/without_error/SKopierer619051610220_0036.bmp",
        "./data/scans_labeled/IMG_8/original_size/with_error_2/SKopierer619051610220_0016.bmp",
        "./data/scans_labeled/IMG_8/original_size/error_mask_2/SKopierer619051610220_0016.png"),
        (
        "./data/scans_labeled/IMG_8/original_size/without_error/SKopierer619051610220_0035.bmp",
        "./data/scans_labeled/IMG_8/original_size/without_error/SKopierer619051610220_0036.bmp",
        "./data/scans_labeled/IMG_8/original_size/with_error_1/SKopierer619051610220_0015.bmp",
        "./data/scans_labeled/IMG_8/original_size/error_mask_1/SKopierer619051610220_0015.png"),
        (
        "./data/scans_labeled/IMG_10/original_size/without_error/SKopierer619051610220_0033.bmp",
        "./data/scans_labeled/IMG_10/original_size/without_error/SKopierer619051610220_0034.bmp",
        "./data/scans_labeled/IMG_10/original_size/with_error_2/SKopierer619051610220_0014.bmp",
        "./data/scans_labeled/IMG_10/original_size/error_mask_2/SKopierer619051610220_0014.png"),
        (
        "./data/scans_labeled/IMG_10/original_size/without_error/SKopierer619051610220_0033.bmp",
        "./data/scans_labeled/IMG_10/original_size/without_error/SKopierer619051610220_0034.bmp",
        "./data/scans_labeled/IMG_10/original_size/with_error_1/SKopierer619051610220_0013.bmp",
        "./data/scans_labeled/IMG_10/original_size/error_mask_1/SKopierer619051610220_0013.png"),
        (
        "./data/scans_labeled/IMG_11/original_size/without_error/SKopierer619051610220_0031.bmp",
        "./data/scans_labeled/IMG_11/original_size/without_error/SKopierer619051610220_0032.bmp",
        "./data/scans_labeled/IMG_11/original_size/with_error_2/SKopierer619051610220_0012.bmp",
        "./data/scans_labeled/IMG_11/original_size/error_mask_2/SKopierer619051610220_0012.png"),
        (
        "./data/scans_labeled/IMG_11/original_size/without_error/SKopierer619051610220_0031.bmp",
        "./data/scans_labeled/IMG_11/original_size/without_error/SKopierer619051610220_0032.bmp",
        "./data/scans_labeled/IMG_11/original_size/with_error_1/SKopierer619051610220_0011.bmp",
        "./data/scans_labeled/IMG_11/original_size/error_mask_1/SKopierer619051610220_0011.png"),
        (
        "./data/scans_labeled/IMG_12/original_size/without_error/SKopierer619051610220_0029.bmp",
        "./data/scans_labeled/IMG_12/original_size/without_error/SKopierer619051610220_0030.bmp",
        "./data/scans_labeled/IMG_12/original_size/with_error_2/SKopierer619051610220_0010.bmp",
        "./data/scans_labeled/IMG_12/original_size/error_mask_2/SKopierer619051610220_0010.png"),
        (
        "./data/scans_labeled/IMG_12/original_size/without_error/SKopierer619051610220_0029.bmp",
        "./data/scans_labeled/IMG_12/original_size/without_error/SKopierer619051610220_0030.bmp",
        "./data/scans_labeled/IMG_12/original_size/with_error_1/SKopierer619051610220_0009.bmp",
        "./data/scans_labeled/IMG_12/original_size/error_mask_1/SKopierer619051610220_0009.png"),
        (
        "./data/scans_labeled/IMG_16/original_size/without_error/SKopierer619051610220_0027.bmp",
        "./data/scans_labeled/IMG_16/original_size/without_error/SKopierer619051610220_0028.bmp",
        "./data/scans_labeled/IMG_16/original_size/with_error_2/SKopierer619051610220_0008.bmp",
        "./data/scans_labeled/IMG_16/original_size/error_mask_2/SKopierer619051610220_0008.png"),
        (
        "./data/scans_labeled/IMG_16/original_size/without_error/SKopierer619051610220_0027.bmp",
        "./data/scans_labeled/IMG_16/original_size/without_error/SKopierer619051610220_0028.bmp",
        "./data/scans_labeled/IMG_16/original_size/with_error_1/SKopierer619051610220_0007.bmp",
        "./data/scans_labeled/IMG_16/original_size/error_mask_1/SKopierer619051610220_0007.png"),
        (
        "./data/scans_labeled/IMG_20/original_size/without_error/SKopierer619051610220_0023.bmp",
        "./data/scans_labeled/IMG_20/original_size/without_error/SKopierer619051610220_0024.bmp",
        "./data/scans_labeled/IMG_20/original_size/with_error_2/SKopierer619051610220_0004.bmp",
        "./data/scans_labeled/IMG_20/original_size/error_mask_2/SKopierer619051610220_0004.png"),
        (
        "./data/scans_labeled/IMG_20/original_size/without_error/SKopierer619051610220_0023.bmp",
        "./data/scans_labeled/IMG_20/original_size/without_error/SKopierer619051610220_0024.bmp",
        "./data/scans_labeled/IMG_20/original_size/with_error_1/SKopierer619051610220_0003.bmp",
        "./data/scans_labeled/IMG_20/original_size/error_mask_1/SKopierer619051610220_0003.png"),
        (
        "./data/scans_labeled/IMG_25/original_size/without_error/SKopierer619051610220_0021.bmp",
        "./data/scans_labeled/IMG_25/original_size/without_error/SKopierer619051610220_0022.bmp",
        "./data/scans_labeled/IMG_25/original_size/with_error_2/SKopierer619051610220_0002.bmp",
        "./data/scans_labeled/IMG_25/original_size/error_mask_2/SKopierer619051610220_0002.png"),
        (
        "./data/scans_labeled/IMG_25/original_size/without_error/SKopierer619051610220_0021.bmp",
        "./data/scans_labeled/IMG_25/original_size/without_error/SKopierer619051610220_0022.bmp",
        "./data/scans_labeled/IMG_25/original_size/with_error_1/SKopierer619051610220_0001.bmp",
        "./data/scans_labeled/IMG_25/original_size/error_mask_1/SKopierer619051610220_0001.png"),
        ]
class DigitalPrintingDefects(Dataset):

    def __init__(self, image, error=2, dual=True, reverse_reference=False, mode="rgb", patch_size=224, stride=200, transform=transforms.ToTensor(), z_normalize=False, balanced_test_set=True):
        super().__init__()

        self.image=image 
        self.error=error 
        self.dual=dual 
        self.mode=mode
        self.patch_size=patch_size 
        self.stride=stride 
        self.transform=transform

        if not "IMG_" in image:
            raise ValueError("Image must start with IMG_<NUMBER>")

        match = False
        for im_files in FILES:
            reference1, reference2, err, mask = im_files
            if image in reference1 and "with_error_" + str(error) in err:
                match = True
                break
        if match == False:
            raise ValueError("No corresponding image found..")

        print("Loading train data.")
        if reverse_reference:
            reference = [str(NasFile(server_path=reference2))]
            if dual is True:
                reference.append(str(NasFile(server_path=reference1)))
        else:
            reference = [str(NasFile(server_path=reference1))]
            if dual is True:
                reference.append(str(NasFile(server_path=reference2)))

        # Normalize the downloaded masks to have only 3 channels      
        e = str(NasFile(server_path=err))
        m = str(NasFile(server_path=mask))
        files = glob.glob("./**/DigitalPrinting/semih/**/*.png", recursive=True)
        for f in files:
            i = imageio.imread(f)
            if i.ndim == 3:
                print(f"Mask {f} has 3 channels. Only keep first.")
                imageio.imsave(f, i[:,:,0])

        train = ImagePatches(reference,
                            mode=mode, 
                            train_percentage=1.0,
                            train=True, 
                            stride_y=stride,
                            stride_x=stride, 
                            window_size=patch_size, 
                            transform=transform,
                            limit=-1,
                            shuffle=False,
                            crop=[5, 5, -1, -1])
        
        data_train = [train[i][0] for i in range(len(train))]
        data_train = torch.cat(data_train).reshape(len(train), -1).numpy()
       
        print("Loading inlier data.")
        test_inliers = LabeledImagePatches(str(NasFile(server_path=err)),
                                                str(NasFile(server_path=mask)),
                                                mode=mode, 
                                                train=True,
                                                oneclass=True,
                                                transform=transform,
                                                train_percentage=1.0,
                                                stride_y=patch_size//2,
                                                stride_x=patch_size//2, 
                                                window_size=patch_size, 
                                                limit=-1,
                                                crop=[1, 1, -1, -1],
                                                anomaly_offset_percentage=95,
                                                shuffle=False)

        data_test_inliers = [test_inliers[i][0] for i in range(len(test_inliers))]
        data_test_inliers = torch.cat(data_test_inliers).reshape(len(test_inliers), -1).numpy()
        
        print("Loading outliers data.")
        test_outliers = LabeledImagePatches(str(NasFile(server_path=err)),
                                                str(NasFile(server_path=mask)),
                                                mode=mode, 
                                                train=False,
                                                oneclass=True,
                                                transform=transform,
                                                train_percentage=1.0,
                                                stride_y=patch_size//2,
                                                stride_x=patch_size//2, 
                                                window_size=patch_size, 
                                                limit=-1,
                                                crop=[1, 1, -1, -1],
                                                anomaly_offset_percentage=95,
                                                shuffle=False)

        data_test_outliers = [test_outliers[i][0] for i in range(len(test_outliers))]
        data_test_outliers = torch.cat(data_test_outliers).reshape(len(test_outliers), -1).numpy()

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

        # Provide Design Matrices as members
        self.test_inliers = test_inliers
        self.test_outliers = test_outliers
        self.train = train
        if z_normalize:
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
        return "{}(image={}, error={}, dual={}, mode={}, patch_size={}, stride={}, data_train={}, data_test_inliers={}, data_test_outliers={})".format(
            self.__class__.__name__,
            self.image, self.error ,self.dual ,self.mode,self.patch_size ,self.stride, self.data_train.shape, self.data_test_inliers.shape, self.data_test_outliers.shape)

if __name__ == "__main__":
    data_train, data_test_inliers, data_test_outliers = DigitalPrintingDefects("IMG_20", error=1)[0]
    data_train, data_test_inliers, data_test_outliers = DigitalPrintingDefects("IMG_20", error=2)[0]