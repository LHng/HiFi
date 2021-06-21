from __future__ import print_function, division
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms
import os

class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, string_name = sample['image_x'],sample['binary_mask'],sample['string_name']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'string_name': string_name}

class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, string_name = sample['image_x'],sample['binary_mask'],sample['string_name']
        image_x = image_x[:, :, ::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
                        
        binary_mask = np.array(binary_mask)
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'string_name': string_name}

class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)

        image_x, binary_mask = self.get_single_image_x(image_path)
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'string_name': videoname}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path):
        binary_mask = np.zeros((32, 32))
        image_x_temp = cv2.imread(image_path)
        image_x_temp_gray = cv2.imread(image_path, 0)
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        image_x= cv2.resize(image_x_temp, (256, 256))

        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[ i, j]=1.0
                else:
                    binary_mask[i, j]=0.0

        return image_x, binary_mask
