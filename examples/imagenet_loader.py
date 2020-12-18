import glob
import imageio
import torchvision
import numpy

class Imagenet():
    
    def __init__(self, root="/mnt/ds1/ILSVRC2015/Data/CLS-LOC/", dataset="train",
                       transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), limit=-1): 

        self.size = 200 // 2
        self.root = root
        
        if dataset == "train":
            self.paths = glob.glob(root + "train/**/**.JPEG") 
        if dataset == "test":
            self.paths = glob.glob(root + "test/**.JPEG") 
        if dataset == "val":
            self.paths = glob.glob(root + "val/**.JPEG") 
   
        self.transforms = transforms
        
        if limit > 0:
            self.paths = self.paths[:limit]      

    def __iter__(self):
        self.a = -1
        return self
    
    def __next__(self):
        self.a += 1
        if self.a >= len(self):
            raise StopIteration()        
        return self[self.a]
    
    def __getitem__(self, i):
        x = imageio.imread(self.paths[i])
        
        ch, cw = x.shape[0]//2, x.shape[1]//2
        if len(x.shape) == 3:
            x = x[ch-self.size:ch+self.size,cw-self.size:cw+self.size,:]
        else:
            x = x[ch-self.size:ch+self.size,cw-self.size:cw+self.size, None]    
        x = x - x.mean()
        
        x_ = np.zeros((200, 200, 3))
        x_[:x.shape[0], :x.shape[1], :] = x
        
        if self.transforms:
            x_ = self.transforms(x_)
            
        return x_.float().view(-1), -1
        
    def __len__(self):
        return len(self.paths)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(root={self.root})"