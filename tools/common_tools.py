# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : JLChen
# @date       : 2021-06
# @brief      : 通用函数库
"""

import os



def set_up(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

