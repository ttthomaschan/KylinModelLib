"""
# @file name  : det_infer.py
# @author     : JLChen
# @date       : 2021-07
# @brief      : DBNet inference
"""

import os
import sys
import pathlib

# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchvision import transforms
from architectures.det_Model import DetModel
from postprocess.det_DBpostprocess import DBPostProcess

from addict import Dict

config = Dict()
config.model = {
    'type': "DetModel",
    'backbone': {"type": "ResNet", 'layers': 18, 'pretrained': False}, # ResNet or MobileNetV3
    'neck': {"type": 'DB_fpn', 'out_channels': 256},
    'head': {"type": "DBHead"},
    'in_channels': 3,
}
config.post_process = {
    'type': 'DBPostProcess',
    'thresh': 0.3,  # 二值化输出map的阈值
    'box_thresh': 0.7,  # 低于此阈值的box丢弃
    'unclip_ratio': 1.5  # 扩大框的比例
}

class ResizeFixedSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) :
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']
        h, w, _ = im.shape
        if min(h, w) < self.short_size:
            if h < w:
                ratio = float(self.short_size) / h
            else:
                ratio = float(self.short_size) / w
        else:
            ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            import sys
            sys.exit(0)

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        if self.resize_text_polys:
            text_polys[:, 0] *= ratio_h
            text_polys[:, 1] *= ratio_w

        data['img'] = img
        data['text_polys'] = text_polys
        return data


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        state_dict = ckpt['state_dict']
        self.model = DetModel(config.model)
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.resize = ResizeFixedSize(736, False)
        self.post_process = DBPostProcess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        # 数据放入模型
        out = self.model(tensor)
        out_np = out.detach().cpu().numpy()
        ## 后处理-对模型结果进行处理
        box_list, score_list = self.post_process(out_np, data['shape'], is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list

if __name__ == '__main__':
    import cv2

    def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
        import cv2
        if isinstance(img_path, str):
            img_path = cv2.imread(img_path)
            # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
        img_path = img_path.copy()
        for point in result:
            point = point.astype(int)
            cv2.polylines(img_path, [point], True, color, thickness)
        return img_path

    img_path = "/home/elimen/Data/dbnet_pytorch/test_images/ins_photo/1.中华联合-金旺交强险.jpeg"
    model_path = "/home/elimen/Data/dbnet_pytorch/checkpoints/ch_det_server_db_res18.pth"
    img = cv2.imread(img_path)
    img_bak = img.copy()
    model = DetInfer(model_path)
    box_list, score_list = model.predict(img, is_output_polygon=False)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_bbox(img, box_list)
    imageres_path = './'
    imageres_name = 'test_result.jpg'
    cv2.imwrite(imageres_path + imageres_name, img)