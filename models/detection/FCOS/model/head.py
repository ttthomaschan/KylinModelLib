import torch
import torch.nn as nn
import math

class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        