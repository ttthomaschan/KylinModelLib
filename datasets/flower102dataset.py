# -*- coding: utf-8 -*-
"""
# @file name  : flower102dataset.py
# @author     : JLChen
# @date       : 2021-06
# @brief      : flower-102 数据集读取
"""

import os 
import random
from PIL import Image
from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        获取数据集的路径、预处理的方法
        '''
        self.cls_num = 102
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path,label), ... , ]
        self.label_array = None
        self._get_img_info()
        

    def __getitem__(self, index):
        '''
        输入标量index，从硬盘中读取数据，并做预处理，以及转换（to Tensor）
        :param index:
        :return:
        '''
        # print("__getitem__() is called.")  ## ## ==> 当调用 实例对象[] 加索引时，就会自动调用此函数
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        '''
        返回数据集长度
        '''
        # print("__len__() is called.")  ## ==> 当调用len(实例对象)时，就会自动调用此函数
        return len(self.img_info)

    def __str__(self):
        # print("__str__() is called.")  ## ==> 当调用 实例对象 时，就会自动调用此函数
        return "Used for description."

    def _get_img_info(self):
        '''
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path,label
        '''
        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith(".jpg")]

        # 读取mat形式label
        label_file = "imagelabels.mat" # hard code
        path_label_file = os.path.join(self.root_dir, "..", label_file)
        from scipy.io import loadmat
        label_array = loadmat(path_label_file)['labels'].squeeze()
        self.label_array = label_array

        # 匹配label
        idx_imgs = [int(n[6:11]) for n in names_imgs]

        path_imgs = [os.path.join(self.root_dir,n) for n in names_imgs]
        self.img_info = [(p,int(label_array[idx-1]-1)) for p, idx in zip(path_imgs, idx_imgs)]

if __name__ == "__main__":
    
    root_dir = r"/home/elimen/Data/deepshare/Classification/102flowers/train"
    test_dataset = FlowerDataset(root_dir)

    print(test_dataset)
    print(test_dataset[1])
    print(len(test_dataset))
    print(next(iter(test_dataset)))