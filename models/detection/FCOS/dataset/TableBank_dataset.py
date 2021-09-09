from torchvision.datasets import CocoDetection
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import os
import cv2
import random
import json
from sklearn.model_selection import train_test_split


class TablebankDataset(Dataset):
    CLASSES_NAME = ('__back_ground__', 'table')
    def __init__(self,imgs_path, anno_path, resize_size=[800,1333], is_train = True, transform=None):
        super().__init__()
        print("INFO====>check annos, filtering invalid data......")

        self.imgs_path = imgs_path
        ## 读入annotations文件，并分别取出label部分和image部分
        with open(anno_path, 'r') as ann_f:
            ann_dict = json.load(ann_f)
        self.anno_dict = ann_dict['annotations']
        self.img_dict = ann_dict['images']

        self.imgid2annoidx = {}
        for i in range(len(self.anno_dict)):
            if str(self.anno_dict[i]['image_id']) in self.imgid2annoidx.keys():
                self.imgid2annoidx[str(self.anno_dict[i]['image_id'])].append(i)
            else:
                self.imgid2annoidx[str(self.anno_dict[i]['image_id'])] = [i]

        self.boxCountMax = 0
        for _, v in self.imgid2annoidx.items():
            self.boxCountMax = max(self.boxCountMax, len(v))

        self.imgdict_train, self.imgdict_val = train_test_split(self.img_dict, test_size=0.1, random_state=22)

        self.category2id = {'table': 1}
        self.id2category = {'1': "table"}

        self.transform = transform
        self.resize_size = resize_size

        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]
        self.train = is_train
        print("call TableBank_dataset.__init__().")

    def __getitem__(self,  index):

        if self.train:
            img_id = self.imgdict_train[index]['id']
            img_file = self.imgdict_train[index]['file_name']
        else:
            img_id = self.imgdict_val[index]['id']
            img_file = self.imgdict_val[index]['file_name']

        anno_index_list = self.imgid2annoidx[str(img_id)]
        img = cv2.imread(os.path.join(self.imgs_path, img_file))
        boxes = []
        classes = []
        for i in range(len(anno_index_list)):
            boxes.append(self.anno_dict[anno_index_list[i]]['bbox'])
            classes.append(self.anno_dict[anno_index_list[i]]['category_id'])
        # list --> ndarray
        boxes = np.array(boxes, dtype = np.float32)
        # xywh --> xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]

        # if self.train:
        #     if random.random() < 0.5 :
        #         img, boxes = flip(img, boxes)
        #     if self.transform is not None:
        #         img, boxes = self.transform(img, boxes)
        # img = np.array(img)

        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
        # img = draw_bboxes(img,boxes)


        img = transforms.ToTensor()(img)
        # img = transforms.Normalize(self.mean, self.std,inplace=True)(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def __len__(self):
        if self.train:
            return len(self.imgdict_train)
        else:
            return len(self.imgdict_val)

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h,  w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side/smallest_side
        if largest_side*scale > max_side:
            scale = max_side/largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw%32
        pad_h = 32 - nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(torch.nn.functional.pad(img, (0, int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))

        max_num=0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num : max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i], (0,0,0,max_num-boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i], (0,max_num-classes_list[i].shape[0]), value=-1))


        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes



if __name__=="__main__":

    dataset = TablebankDataset("/home/elimen/Data/OCR_dataset/Table/TableBank/Detection/images", "/home/elimen/Data/OCR_dataset/Table/TableBank/Detection/annotations/tablebank_word_train.json")
    # img, boxes, classes = dataset[0]  ## 直接给索引，调用的是forward()
    # cv2.imwrite("./123.jpg",img)  ## error，此时img格式为torch.Tensor
    img, boxes, classes = dataset.collate_fn([dataset[0], dataset[1], dataset[2]])  ## 此处直接调用的collate_fn()

    print(boxes, "\n", classes, "\n", img.shape, boxes.shape, classes.shape, "\n", boxes.dtype, classes.dtype, img.dtype)
    print("Done.")