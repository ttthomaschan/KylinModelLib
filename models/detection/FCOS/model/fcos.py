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

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>Success frozen BN.")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>Success frozen backbone stage1")

class DetectHead(nn.Module):
    def __init__(self, score_threshold,
                 nms_iou_threshold,
                 max_detection_boxes_num,
                 strides,
                 config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            config = DefaultConfig
        else:
            config = config

    def _reshape_cat_out(self, inputs, strides):
        '''
        inputs: list contains five [batch_size, c, _h, _w]
        Returns
        out: [batch_size, sum(_h*_w), c]
        coords [sum(_h*_w), 2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0,2,3,1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)

    def _coords2boxes(self, coords, offsets):
        '''
        coords: [sum(_h*_w), 2]
        offsets: [batch_size, sum(_w*_h), 4]
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def batched_nms(self, boxes, scores, idxs, iou_threshold):


    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords =