"""
# @file name  : det_Model.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : DBNet Model
"""

print(__file__)
import os
import sys
import pathlib
# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))
print(str(__dir__.parent.parent))
from backbones.det_resnet import ResNet
from necks.det_fpn import DB_fpn
from heads.det_DBhead import DBHead
import torch
from torch import nn
from addict import Dict

config = Dict()
config.model = {
    'type': "DetModel",
    'backbone': {"type": "ResNet", 'layers': 18, 'pretrained': True}, # ResNet or MobileNetV3
    'neck': {"type": 'DB_fpn', 'out_channels': 256},
    'head': {"type": "DBHead"},
    'in_channels': 3,
}


class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet(config['in_channels'],config['backbone']['layers'])
        self.neck = DB_fpn(self.backbone.out_channels)
        self.head = DBHead(self.neck.out_channels)

        self.name = f'DetModel_DBNet'

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = DetModel(config.model)
    state_dict_path = '/home/junlin/Git/github/dbnet_pytorch/checkpoints/ch_det_server_db_res18.pth'
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict['state_dict'])
    print(next(model.head.parameters()))