import torch
import torch.utils.data
import numpy as np


class SSDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dset, opt):
        self.dset = dset
        self.opt = opt
        
    def __getitem__(self, index):
        image, target, item = self.dset[index]
        
        if(not(self.opt.ssl)):
            return image, target, item
        else:
            if(self.opt.ssl_rot):
                label = np.random.randint(4)
                if label == 1:
                    image_rot = tensor_rot_90(image)
                elif label == 2:
                    image_rot = tensor_rot_180(image)
                elif label == 3:
                    image_rot = tensor_rot_270(image)
                else:
                    image_rot = image
                    
                return (image, image_rot), (target, label), item
            
            if(self.opt.ssl_quad):
                label = np.random.randint(4)

                horstr = image.size(1) // 2
                verstr = image.size(2) // 2
                horlab = label // 2
                verlab = label % 2

                image_quad = image[:, horlab*horstr:(horlab+1)*horstr, verlab*verstr:(verlab+1)*verstr,]
                return (image, image_quad), (target, label), item
            
    def __len__(self):
        return len(self.dset)
                
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1,2)
def tensor_rot_90_digit(x):
	return x.transpose(1,2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)
def tensor_rot_180_digit(x):
	return x.flip(2)

def tensor_rot_270(x):
	return x.transpose(1,2).flip(2)