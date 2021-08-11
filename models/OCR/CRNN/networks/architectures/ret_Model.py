"""
# @file name  : rec_Model.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : CRNN Model
"""

print(__file__)
import os
import sys
import pathlib
# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))
print(str(__dir__.parent.parent))
from backbones.rec_resnet import ResNet
from necks.rec_bilstm import SequenceEncoder
from heads.rec_ctchead import CTC
import torch
from torch import nn
from addict import Dict

config = Dict()
config.model = {
    'type': "RecModel",
    'backbone': {"type": "ResNet", 'layers': 34},
    'neck': {"type": 'rnn'},
    'head': {"type": "CTC", 'n_class': 6625},
    'in_channels': 3,
}

class RecModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet(config['in_channels'], config['backbone']['layers'])
        self.neck = SequenceEncoder(self.backbone.out_channels)
        self.head = CTC(self.neck.out_channels, config['head']['n_class'])

        self.name = f'RecModel_CRNN'

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

if __name__  == "__main__":
    model = RecModel(config.model)
    state_dict_path = '/home/elimen/Data/dbnet_pytorch/checkpoints/ch_rec_server_crnn_res34.pth'
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict['state_dict'])
    print('Model loaded.')

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    