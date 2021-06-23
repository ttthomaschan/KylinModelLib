# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : JLChen
# @date       : 2021-06
# @brief      : 
"""

import torch
import numpy as np
from collection import Counter

class ModelTrainer(object):
    
    @staticmethod
    def train(data_loader,model,loss_f,optimizer,scheduler,epoch_idx,device,log_interval,max_epoch):
        model.train()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num,class_num)))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        label_list = []

        for i, data in enumerate(dataloader):
            
            _, label, _ = data
            label_list.expend(label.tolist())

            inputs, labels, path_imgs = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(),labels.cpu())
            loss.backward()
            optimizer.step()

            # loss statistics
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # Print every 10 iterations
            if i%log_interval == log_interval -1:
                print("Training: Epoch[{}/{}] Iteration[{}/{}] Loss:{} Acc:{}".format(epoch_idx+1,max_epoch,i+1,len(data_loader),loss_mean,acc_avg))
        
        print("epoch:{} sampler:{}".format(epoch_idx, Counter(label_list)))
        return loss_mean, acc_avg, conf_mat, path_error