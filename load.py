from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import random


class mri_dataset(Dataset):
    def __init__(self, file_dir, transform=None):
        self.file_dir = file_dir
        self.file_list = os.listdir(file_dir)
        self.len = len(self.file_list)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.file_list[index]
        # k_array[0] - real / k_array[1] - imag
        k_array = np.load(os.path.join(self.file_dir,filename))
        # randomly choose btw 4-fold or 8-fold
        flg = random.choice([0,1])
        if flg == 0:
            un_k_array = undersample(k_array, 4)
        elif flg == 1:
            un_k_array = undersample(k_array, 8)
        
        # 3d tensor to 4d tensor
        if self.transform:
            img = self.transform(un_k_array)
            ground_truth = self.transform(k_array)
        elif self.transform==None:
            img = un_k_array
            ground_truth = k_array
         
        return (img, ground_truth)

    def __len__(self):
        return self.len