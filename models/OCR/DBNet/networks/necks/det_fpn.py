"""
# @file name  : det_fpn.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : DBNet neck -- FPN
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DB_fpn(nn.Module):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super().__init__()
        inplace = True
        self.out_channels = out_channels
        # reduce layers using 1*1 conv-filter
        self.in2_conv = nn.Conv2d(in_channels[0], self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels[1], self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels[2], self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels[3], self.out_channels, kernel_size=1, bias=False)
        # Smooth layers
        self.p5_conv = nn.Conv2d(self.out_channels, self.out_channels//4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(self.out_channels, self.out_channels//4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(self.out_channels, self.out_channels//4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(self.out_channels, self.out_channels//4, kernel_size=3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        p3 = F.interpolate(p3, scale_factor=2)
        p4 = F.interpolate(p4, scale_factor=4)
        p5 = F.interpolate(p5, scale_factor=8)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def forward(self, x):
        # 从backbone传入不同尺寸特征图
        c2, c3, c4, c5 = x
        # 减少通道数，使每个尺寸通道数一致
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)
        # 先上采样再相加，融合相邻两层特征层
        out4 = self._upsample_add(in5, in4)
        out3 = self._upsample_add(out4, in3)
        out2 = self._upsample_add(out3, in2)
        # 平滑特征层，调整输出层数
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        # 拼接4层
        x = self._upsample_cat(p2, p3, p4, p5)
        return x
