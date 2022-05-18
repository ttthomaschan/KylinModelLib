import torch
from mmcv import Registry
from mmcv.utils.registry import build_from_cfg

CLASS = Registry("class_test")


class aclass(object):
    def __init__(self):
        self.color = "red"

