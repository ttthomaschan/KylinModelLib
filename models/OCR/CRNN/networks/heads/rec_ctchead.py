# encoding: utf-8
"""
# @file name  : rec_ctc.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : CRNN head
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    net = CTC(1024, 11)
    print(net.state_dict().keys())