import torch
import torch.nn as nn
from .config import DefaultConfig

def coords_fmap2orig(feature, stride):
    '''
    transform one feature map coords to original img coords.
    :param feature: [batch_size, h, w, c]
    :param stride: int --> downsample factor
    :return: coords: [n,2]
    '''
    h, w = feature.shape[1:3]
    shifts_x = torch.arrange(0, w*stride, stride, dtype=torch.float32)
    shifts_y = torch.arrange(0, h*stride, stride, dtype=torch.float32)

    # meshgrid()生成网格，可以用于生成坐标。
    # 函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数。
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)  ## y为行，x为列
    shifts_x = torch.reshape(shifts_x, [-1])
    shifts_y = torch.reshape(shifts_y, [-1])
    coords = torch.stack([shifts_x, shifts_y], -1) + stride//2
    return coords

class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        '''

        :param inputs:
        :return:
        '''