import torch
import torch.nn as nn

from .config import DefaultConfig

from .backbone.resnet import resnet50
from .neck import FPN
from .head import ClsCntRegHead
# from .loss import

class FCOS(nn.Module):
    def __init__(self, config=None):
        super(FCOS, self).__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet50(pretrain=config.pretrained, if_include_top=False)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels,
                                  config.class_num,
                                  config.use_GN_head,
                                  config.cnt_on_reg,
                                  config.prior)
        self.config = config

    def forward(self, x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]

    def train(self, mode=True):
        '''set module training mode and freeze bn'''
        super().train(mode=True)

