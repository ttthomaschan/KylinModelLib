# -*- coding: utf-8 -*-
"""
# @file name  : flower102dataset.py
# @author     : JLChen
# @date       : 2021-06
# @brief      : 模型训练主代码
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from datasets.flower102dataset import FlowerDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
import torch.nn as nn
import torch.optim as optim
from tools.model_trainer import ModelTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    # step0： 参数配置
    train_dir = "/home/elimen/Data/deepshare/Classification/102flowers/train"
    valid_dir = "/home/elimen/Data/deepshare/Classification/102flowers/valid"
    pretraided_model_path = os.path.join(BASE_DIR, "../pretrained_model","resnet18-5c106cde.pth")

    norm_mean = []  # ImageNet 120万图像中统计出来的各通道均值和方差
    norm_std = []
    normTransform = transforms.Normalize(norm_mean, norm_std)

    transforms_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normTransform
    ])

    transforms_valid = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normTransform
    ])

    train_bs = 128
    valid_bs = 128
    workers = 8

    lr_init = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    factor = 0.1
    milestones = [30,45]
    max_epoch = 50
    

    # step1： 数据集
    # 构建 MyDataset 实例
    train_data = FlowerDataset(root_dir=train_dir,transform=transforms_train)
    valid_data = FlowerDataset(root_dir=valid_dir,transform=transforms_valid)
    # 构建 DataLoader
    train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True,num_workers=workers)
    valid_loader = DataLoader(dataset=valid_data,batch_size=valid_bs,shuffle=False,num_workers=workers)
    
    # step2： 模型
    model = resnet18(pretrained=True)

    # 导入预训练模型
    if os.path.exists(pretraided_model_path):
        pretrained_state_dict = torch.load(pretraided_model_path, map_location='cpu')
        model.load_state_dict(pretrained_state_dict)
    # 修改输出层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.cls_num)
    model.to(device)

    # step3： 损失函数、优化器
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=lr_init,momentum=momentum,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,gamma=factor,milestones=milestones)

    # step4： 迭代训练
    for epoch in range(max_epoch):
        for 