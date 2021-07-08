# -*- coding: utf-8 -*-
"""
# @file name  : flower102dataset.py
# @author     : JLChen
# @date       : 2021-06
# @brief      : VOC 2012 数据集读取
"""

import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
import math

class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, root_dir, resize_size=[800, 1333], split='trainval', use_difficult=False, is_train=True,
                 augment=None):
        """初始化主要工作：
            0）告知文件路径
            1）self.img_ids 图片名称列表
            2）self.name2id 标签类别到整数的映射字典
        """
        self.root = root_dir
        self.imgset = split
        self.use_difficult = use_difficult

        # %s 字符串格式符 字符串作为模板。
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.imgset) as f:
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]

        self.name2id = dict(zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}

        self.resize_size = resize_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train = is_train
        self.augment = augment

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        img_id = self.img_ids[index]
        print(img_id)

        img_path = self._imgpath % img_id
        print(img_path)
        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # 读入rgb图像，，，
        img = Image.open(img_path)  # ，，，同上
        plt.imshow(img)

        label_path = self._annopath % img_id
        print(label_path)
        boxes, classes = self.get_xml_label(label_path)

        # 可视化一下：
        current_axis = plt.gca()
        for i, box in enumerate(boxes):
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            label = classes[i]
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='yellow', fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='red', bbox={'facecolor': 'white', 'alpha': 0.6})

            # 数据格式转换  都转换成张量
        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img.shape, boxes, classes

    def get_xml_label(self, label_path):

        anno = ET.parse(label_path).getroot()
        boxes = []
        classes = []
        for obj in anno.iter("object"):
            # 放弃难分辨的图片
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue

            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            # 1 对一下 这个框像素点位置
            TO_REMOVE = 1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)
            # 2 找一下这个框的类别码
            name = obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])

        boxes = np.array(boxes, dtype=np.float32)
        return boxes, classes


if __name__ == "__main__":
    dataset = VOCDataset("/home/elimen/Data/deepshare/Object Detection/课件完全版/VOCdevkit/VOC2012", split='trainval_demoData')
    dataset[3]