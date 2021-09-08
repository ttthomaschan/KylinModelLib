# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : JLChen
# @date       : 2021-06
# @brief      : 通用函数库
"""

import os
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from torchvision.models import resnet18


def setup_seed(seed=2938):
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)   # cpu

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True    # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法

def check_dir(path_tmp):
    assert os.path.exists(path_tmp), \
        "\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath(path_tmp))

class Logger(object):
    def __init__(self, log_path):
        log_name = os.path.basename(log_path)  ## 截取文件名
        log_dir = os.path.dirname(log_path)    ## 截取路径
        self.log_name = log_name if log_name else "root"
        self.out_path = log_path

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 添加Handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

def make_logger(out_dir):
    '''
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger实例，用于记录信息
    :param out_dir:  str
    :return:
    '''
    now_time = datetime.now()
    time_str = datetime.strftime(now_time,"%m-%d_%H-%M")
    log_dir = os.path.join(out_dir, time_str)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # 创建logger
    log_path = os.path.join(log_dir, "log.log")
    logger = Logger(log_path)
    logger = logger.init_logger()
    return logger, log_dir


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    '''
    绘制训练集和验证集的 loss曲线/acc曲线
    :param train_x:
    :param train_y:
    :param valid_x:
    :param valid_y:
    :param mode:  "loss" or "acc"
    :param out_dir:
    :return:
    '''
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()



def show_confMat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, percentage=False):
    '''
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes:  list or tuple, 类别名称
    :param set_name:  str, 数据集类型 train/valid/test
    :param out_dir:  str, 图片保存文件夹
    :param epoch:  int, 第几个epoch
    :param verbose:  bool， 是否打印精度信息
    :param percentage: bool, 是否采用百分比【一般图像分割时使用，因各分类数目大】
    :return: 
    '''

    cls_num = len(classes)
    confusion_mat_tmp = confusion_mat.copy()

    # 逐行归一化
    for i in range(cls_num):
        confusion_mat_tmp[i,:] = confusion_mat[i,:]/confusion_mat[i,:].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6,30,91)[cls_num-10]
    plt.figure(figsize=(int(figsize),int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色： http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    xlocation = np.array(range(cls_num))
    plt.xticks(xlocation, list(classes), rotation=60)
    plt.yticks(xlocation, list(classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion_Matrix_{}_{}'.format(set_name, epoch))

    # 打印数字
    if percentage:
        cls_per_nums = confusion_mat.sum(axis=1).reshape((cls_num, 1))
        conf_mat_perc = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_perc[i,j]),
                         va='center', ha='center',
                         color='red', fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(confusion_mat[i,j]),
                         va='center', ha='center',
                         color='red', fontsize=10)

    # 保存
    plt.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total_num:{:<6}, correct_num:{:<5} | Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i,:]), confusion_mat[i,i],
                confusion_mat[i,i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i,i] / (.1 + np.sum(confusion_mat[:, i]))))


