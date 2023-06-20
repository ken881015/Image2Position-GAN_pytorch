import os
import numpy as np
import torch
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
import time
import sys

from Utils import get_landmarks, plot_landmarks_edge

__all__ = [
    "I2P_train",
    "I2P_bm",
    "I2P_bm_re",
    "CelebA",
    "Normalize_wk",
    "ChannelScale_wk",
    "RandomErase_wk",
    "npy2tensor_wk",
    "tensor2npy_wk",
    # "rotate_PIL_wk"
]


class I2P_train(Dataset):
    '''
    - Correct Dataset name: 300W-LP
    training_data_path contains these sub-folders:
        - angle_npy
        - InputImage
        - LabelImage
        - LabelImage_npy
    '''
    def __init__(self, in_dir, transform=None, target_transform=None):
        
        self.images_dir = in_dir + 'InputImage/'
        if not os.path.exists(self.images_dir):
            raise Exception("images_dir does not exist")
        
        self.uvmaps_npy_dir = in_dir + 'LabelImage_npy/'
        if not os.path.exists(self.uvmaps_npy_dir):
            raise Exception("uvmaps_npy_dir does not exist")
        
        self.angle_dir = in_dir + 'angle_npy/'
        if not os.path.exists(self.angle_dir):
            raise Exception("angle_dir does not exist")
        
        self.transform = transform
        self.target_transform = target_transform
        self.data_num = len(os.listdir(self.images_dir))
    
    def __getitem__(self, index):
        if index > self.__len__():
            raise Exception("Index out of range")
        else:
            image = Image.open(self.images_dir + os.listdir(self.images_dir)[index]) # type : PIL
            # Avoiding the file order in two dir are different
            # uvmap = Image.open(self.uvmaps_dir + os.listdir(self.images_dir)[index].replace('.','_posmap.'))
            uvmap_npy = np.load(self.uvmaps_npy_dir + os.listdir(self.images_dir)[index].replace('jpg','npy'))
            roll_angle = np.load(self.angle_dir+ os.listdir(self.images_dir)[index].replace('.jpg','_angle.npy'))
            
            # roll_augments
            if np.abs(roll_angle) < 60: # check the roll angle range in [-30, 30]
                if np.random.rand() < 0.3: # 30% change to be augmented
                    img_w, img_h = image.size
                    angle = 60 + np.random.rand() * 60 # try to move the roll angle from [0,60] to [60, 180]
                    radian = np.deg2rad(angle)
                    image = image.rotate(angle)
                    R = np.array([
                        [np.cos(radian), np.sin(-radian), 0],
                        [np.sin(radian),  np.cos(radian), 0],
                        [             0,               0, 1]
                    ])
                    
                    uvmap_npy = uvmap_npy - np.array([img_w//2,img_h//2,0])
                    uvmap_npy = np.matmul(uvmap_npy, R)
                    uvmap_npy += np.array([img_w//2,img_h//2,0])
                    
                # if np.random.rand() < 0.1:
                #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
                #     uvmap_npy[:,:,0] = 256 - 1 - uvmap_npy[:,:,0] 
            
            
            
            if self.transform is not None:
                image = self.transform(image)
            
            if self.target_transform is not None:
                uvmap_npy = self.target_transform(uvmap_npy)

            return image, uvmap_npy
    
    def __len__(self):
        return self.data_num

class I2P_bm(DatasetFolder):
    def __init__(self, in_dir, transform=None, target_transform=None):
        
        self.in_dir = in_dir
        self.sub_dir_list = os.listdir(in_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.data_num = len(self.sub_dir_list)
        
    def __getitem__(self, index):
        if index > self.__len__():
            raise Exception("Index out of range")
        # else:
        image = Image.open(self.in_dir + os.path.join(self.sub_dir_list[index],self.sub_dir_list[index]) + "_cropped.jpg") # type : PIL
        # Avoiding the file order in two dir are different
        # uvmap = Image.open(self.uvmaps_dir + os.listdir(self.images_dir)[index].replace('.','_posmap.'))
        
        npy = np.load(self.in_dir + os.path.join(self.sub_dir_list[index],self.sub_dir_list[index]) + "_cropped_uv_posmap.npy")
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            npy = self.target_transform(npy)

        return image, npy
    
    def __len__(self):
        return self.data_num
    
    def __get_name_bbox_kpt__(self, index):
        if index > self.__len__():
            raise Exception("Index out of range")
        
        bbox_info = sio.loadmat(self.in_dir + os.path.join(self.sub_dir_list[index],self.sub_dir_list[index]) + "_bbox_info.mat")
        
        name = self.sub_dir_list[index]
        bbox = bbox_info["Bbox"]
        kpt = bbox_info["Kpt"]
        
        return name, bbox, kpt
    
    def get_idx_by_name(self, name):
        return self.sub_dir_list.index(name)

# re = reannotate
class I2P_bm_re(Dataset):
    def __init__(self, in_dir, transform=None, target_transform=None):
        
        self.in_dir = in_dir
        
        with open(self.in_dir+'AFLW2000-3D_crop.list', 'r') as f:
            self.fname_list = [line[:-5] for line in f] # get rid of '.jpg\n'
            
        self.gt_lmk = np.load(self.in_dir+"AFLW2000-3D-Reannotated.pts68.size256.npy")
        
        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform
            
        self.data_num = self.gt_lmk.shape[0]
        
    def __getitem__(self, index):
        if index > self.__len__():
            raise Exception("Index out of range")
        # else:
        image = Image.open(self.in_dir + "AFLW2000-3D_crop_256/"+ self.fname_list[index]+"_cropped.jpg") # type : PIL
        # Avoiding the file order in two dir are different
        # uvmap = Image.open(self.uvmaps_dir + os.listdir(self.images_dir)[index].replace('.','_posmap.'))
        
        npy = self.gt_lmk[index] # shape (2,68)
        
        if self.transform is not None:
            image = self.transform(image)
        
        # if self.target_transform is not None:
        #     npy = self.target_transform(npy)

        return image, npy
    
    def __len__(self):
        return self.data_num
    
    def __get_name_bbox_kpt__(self, index):
        if index > self.__len__():
            raise Exception("Index out of range")
        
        name = self.fname_list[index]
        kpt = self.gt_lmk[index]
        minx, maxx = np.min(kpt[0, :]), np.max(kpt[0, :])
        miny, maxy = np.min(kpt[1, :]), np.max(kpt[1, :])
        bbox = np.array([[minx, miny],[maxx, maxy]])
        
        return name, bbox, kpt
    
    def get_idx_by_name(self, name):
        return self.fname_list.index(name)

class CelebA(DatasetFolder):
    def __init__(self, in_dir, transform=None, target_transform=None):
        
        self.in_dir = in_dir
        self.sub_dir_list = os.listdir(in_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.data_num = len(self.sub_dir_list)
        
    def __getitem__(self, index):
        if index > self.__len__():
            raise Exception("Index out of range")
        # else:
        image = Image.open(self.in_dir + self.sub_dir_list[index]) # type : PIL
        
        if self.transform is not None:
            image = self.transform(image)

        return image
    
    def __len__(self):
        return self.data_num

class Normalize_wk:
    """ 
    Normalizes a ndarray(H,W,C) to [0, 1]
    """
    def __call__(self, npy):
        return (npy - np.min(npy)) / (np.max(npy) - np.min(npy))

class ChannelScale_wk:
    def __init__(self, min_rate=0.6, max_rate=1.4):
        self.min_rate = min_rate
        self.max_rate = max_rate
        
    def __call__(self, image):
        rate = torch.rand(size=(3,1,1)) * (self.max_rate - self.min_rate) + self.min_rate
        image_aug = image * rate
        image_aug = (image_aug - torch.min(image_aug)) / (torch.max(image_aug) - torch.min(image_aug))
        return image_aug
    
class RandomErase_wk:
    def __init__(self, seed, max_num=4, s_l=0.1, s_h=0.6, v_l=0, v_h=1.0):
        self.seed = seed
        self.max_num = max_num
        self.s_l = s_l
        self.s_h = s_h
        self.v_l = v_l
        self.v_h = v_h
        
    def __call__(self, image):
        [_, image_h, image_w] = image.shape
        
        for _ in range(self.max_num):
            w = np.uint8((np.random.uniform(self.s_l, self.s_h) * image_w))
            h = np.uint8((np.random.uniform(self.s_l, self.s_h) * image_h))
            
            left = np.random.randint(0, image_w + w) - w
            top = np.random.randint(0, image_h + h) - h
            
            mask = torch.zeros((image_h, image_w))
            mask[max(top, 0):min(top + h, image_h), max(left,0):min(left + w, image_w)] = 1
            
            choose = np.random.uniform(0,1)
            
            c = torch.rand(size=(3,1,1)) * (self.v_h - self.v_l) + self.v_l
            if choose < 0.5:
                image = torch.where(mask>0, c, image)
            else:
                image = torch.where(mask>0, image*c, image)
        
        return image

class npy2tensor_wk:
    def __call__(self, npy):
        npy = npy.transpose((2, 0, 1))
        npy = npy.astype("float32") / 255.
        tensor = torch.from_numpy(npy)

        return tensor

class tensor2npy_wk:
    def __call__(self, tensor):
        # tensor = tensor.permute(1, 2, 0)
        tensor *= 255.
        npy = tensor.cpu().detach().numpy().transpose((0,2,3,1))
        return npy

# class rotate_PIL_wk:
#     def __call__(self, angle, pil_img):
#         rotated_pil_img = pil_img.rotate(angle)
#         return rotated_pil_img

if __name__ == '__main__':
    dataset = I2P_train("/home/vlsilab/Dataset/Img2Pos_train/")
    image, npy = dataset[0]
    print(npy.shape)
    # npy = np.load("./AFW_134212_1_0_angle.npy")
    # benchmark = I2P_bm("/home/vlsilab/Dataset/Img2Pos_test/AFLW2000_all-crop/")

    # name, bbox, kpt = benchmark.__get_name_bbox_kpt__(1015)
    # image, npy = benchmark[benchmark.get_idx_by_name("image03871")]
    
    # print(image.size)
    
    # uv_kpt_ind = np.loadtxt('../Image/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
    
    # kpt = get_landmarks(npy, uv_kpt_ind) # shape 68 * 3
    
    # angle = 30
    # radian = np.deg2rad(angle)
    
    # R = np.array([
    #     [np.cos(radian), np.sin(-radian), 0],
    #     [np.sin(radian),  np.cos(radian), 0],
    #     [             0,               0, 1]
    # ])
    # rotate_npy = npy - np.array([128,128,0])
    # rotate_npy = np.matmul(rotate_npy, R)
    # rotate_npy += np.array([128,128,0])
    
    # # rotated_kpt = kpt - np.array([[128,128,0]])
    # # rotated_kpt = np.matmul(rotated_kpt, R)
    # # rotated_kpt = rotated_kpt + np.array([[128,128,0]])
    
    # rotated_image = image.rotate(angle)
    
    # image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # a_npy = 256 - 1 - npy[:,:,0] 
    # b_npy = (npy[:,:,0] - 128) * (-1) + 128
    
    # print((a_npy == b_npy).all())
        
    # image = plt.imread("./image03871.jpg")
    # kpt = sio.loadmat("./image03871.mat")["pt3d_68"][:2,:].T
    # print(kpt.shape)
    # sys.exit()
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(image)
    # ax.get_yaxis().set_ticks([]); ax.get_xaxis().set_ticks([])
    # ax.scatter(npy[::10,::10,0], npy[::10,::10,1])
    # plot_landmarks_edge(kpt, ax)
    
    # plt.savefig("align_by_uvmap.png")