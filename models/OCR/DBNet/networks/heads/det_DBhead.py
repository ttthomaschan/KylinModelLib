"""
# @file name  : det_DBhead.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : DBNet neck -- FPN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, in_channels):
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=3, padding=1, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels//4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=2, stride=2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels//4)

        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels//4, out_channels=1, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = torch.sigmoid(x)

        return x
