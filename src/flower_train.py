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
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    # step0： 参数配置
    now_time = datetime.now()
    time_str = datetime.strftime(now_time,"%m-%d_%H-%M")
    log_dir = os.path.join(BASE_DIR,"..","results",time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_dir = "/home/elimen/Data/deepshare/Classification/102flowers/train"
    valid_dir = "/home/elimen/Data/deepshare/Classification/102flowers/valid"
    pretrained_model_path = "/home/elimen/Data/KylinModelLIB/pretrained_model/resnet18-5c106cde.pth"

    norm_mean = [0.485, 0.456, 0.406]  # ImageNet 120万图像中统计出来的各通道均值和方差
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)

    transforms_train = transforms.Compose([
        transforms.Resize((256)),  # (256, 256) 区别； （256） 最短边256
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform,
    ])
    transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform,
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
    log_interval = 10
    

    # step1： 数据集
    # 构建 MyDataset 实例
    train_data = FlowerDataset(root_dir=train_dir, transform=transforms_train)
    valid_data = FlowerDataset(root_dir=valid_dir, transform=transforms_valid)
    # 构建 DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs, shuffle=False, num_workers=workers)
    
    # step2： 模型
    model = resnet18()

    # 导入预训练模型
    if os.path.exists(pretrained_model_path):
        pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')
        model.load_state_dict(pretrained_state_dict)
    # 修改输出层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.cls_num)
    model.to(device)

    # step3： 损失函数、优化器
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=factor, milestones=milestones)

    # step4： 迭代训练
    loss_rec = {"train":[],"valid":[]}
    acc_rec = {"train":[],"valid":[]}
    best_acc, best_epoch = 0, 0

    for epoch in range(max_epoch):
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
                train_loader, model, loss_f, optimizer, scheduler, epoch, device, log_interval, max_epoch)
        
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device)
        
        print("Summary: Epoch[{:0>3}/{:0>3}] || Train Acc: {:.2%} , Valid Acc:{:.2%} || Train loss:{:.4f} , Valid loss:{:.4f} || LR:{}". \
                    format(epoch + 1, max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                           optimizer.param_groups[0]["lr"]))

        scheduler.step()

        # 模型保存
        if best_acc < acc_valid or epoch == max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc            
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch":epoch,
                          "best_acc":best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == max_epoch -1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)