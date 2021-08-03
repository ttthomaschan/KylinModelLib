"""
# @file name  : flower102dataset.py
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
        # reduce layers
        self.in2_conv = nn.Conv2d()