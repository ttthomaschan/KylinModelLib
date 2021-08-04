"""
# @file name  : det_Model.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : DBNet Model
"""

from networks.backbones.det_resnet import ResNet
from networks.necks.det_fpn import DB_fpn
from networks.heads.det_DBHead import DBHead

class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet
        self.neck = DB_fpn()
        self.head = DBHead()

        self.name = f'DetModel_DBNet'

    def forward(self, x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x