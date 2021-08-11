# -*- coding: utf-8 -*-
# @Time    : 2021/7
# @Author  : JLChen
import os
import sys
import pathlib

# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
sys.path.append(str(__dir__.parent.parent.parent))

import torch
from torch import nn
from networks.architectures.ret_Model import RecModel
from datasets.RecDataset import RecDataProcess
from utils.label_convert import CTCLabelConverter
from addict import Dict

config = Dict()
config.model = {
    'type': "RecModel",
    'backbone': {"type": "ResNet", 'layers': 34},
    'neck': {"type": 'rnn'},
    'head': {"type": "CTC", 'n_class': 6625},
    'in_channels': 3,
}
config.dataset = {
    'alphabet': r'/home/elimen/Data/KylinModelLIB/models/OCR/CRNN/datasets/alphabets/ppocr_keys_v1.txt',
    'train': {
        'dataset': {
            'type': 'RecTextLineDataset',
            'file': r'/home/tthom/storage/DATA/ICDAR2015/recognition/train.txt', #'/home/elimen/Data/OCR_dataset/icdar2015/recognition/train.txt',
            'input_h': 32,
            'mean': 0.5,
            'std': 0.5,
            'augmentation': False,
        }
    }
}

class RecInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = RecModel(config.model)
        state_dict = ckpt['state_dict']
        # for k, v in ckpt['state_dict'].items():
        #     state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(config.dataset['train']['dataset'])
        self.converter = CTCLabelConverter(config.dataset['alphabet'])
        # print(config.dataset['alphabet'])

    def predict(self, img):
        # 预处理根据训练来
        # print("input of rec_infer:" + str(type(img)))
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        # print(out.shape)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt



if __name__ == '__main__':
    import cv2
    import os

    img_path = "/home/elimen/Data/KylinModelLIB/models/OCR/CRNN/tmp.png"
    img_dir = "/home/elimen/Data/KylinModelLIB/models/OCR/DBNet/networks/log/tmp_results"
    model_path = "/home/elimen/Data/dbnet_pytorch/checkpoints/ch_rec_server_crnn_res34.pth"

    file_list = []
    text_list = []
    ## 判断文件数量
    if os.path.isdir(img_dir):
        for root, _, files in os.walk(img_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        file_list.append(img_dir)

    f = open('rec_result.txt', "w")

    for i in range(len(file_list)):
        ## 判断文件类型
        img = cv2.imread(file_list[i])
        img_bak = img.copy()
        model = RecInfer(model_path)
        out = model.predict(img)
        text_list.append(out[0][0])

        f.write(out[0][0])
        f.write("\n")

    f.close()
