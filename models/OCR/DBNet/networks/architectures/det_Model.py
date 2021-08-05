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
# from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir='./log', comment='dbnet')

config = Dict()
config.model = {
    'type': "DetModel",
    'backbone': {"type": "ResNet", 'layers': 18, 'pretrained': False}, # ResNet or MobileNetV3
    'neck': {"type": 'DB_fpn', 'out_channels': 256},
    'head': {"type": "DBHead"},
    'in_channels': 3,
}
config.post_process = {
    'type': 'DBPostProcess',
    'thresh': 0.3,  # 二值化输出map的阈值
    'box_thresh': 0.7,  # 低于此阈值的box丢弃
    'unclip_ratio': 1.5  # 扩大框的比例
}


class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet(config['in_channels'], config['backbone']['layers'])
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
    state_dict_path = '/home/elimen/Data/dbnet_pytorch/checkpoints/ch_det_server_db_res18.pth'
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict['state_dict'])
    print('Model loaded.')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    x = torch.zeros(1, 3, 640, 640)
    x = x.to(device)
    out = model(x)

    # with writer:
    #     writer.add_graph(model, x)
    print('Finish.')
