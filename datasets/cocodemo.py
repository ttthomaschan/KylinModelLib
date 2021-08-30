# %matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import pylab
import os

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

dataDir = '/home/elimen/Data/deepshare/DATABASE/COCO2017'
subDir = 'annotations_trainval2017'
dataType = 'val2017'
annFile = '{}/{}/annotations/instances_{}.json'.format(dataDir, subDir, dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format('\n'.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format('\n'.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

'''读取url格式图片，最方便快捷的方式是使用skimage中的api'''
'''skimage 和 cv2 一样，读入的图片格式为nparray'''
img_io = io.imread(img['coco_url'])
img_cv = cv2.imread(os.path.join(os.path.join(dataDir, dataType), img['file_name']))
plt.axis('off')
plt.imshow(img_io)
# plt.show()

# load and display instance annotations
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

# initialize COCO api for person keypoints annotations
annFile = '{}/{}/annotations/person_keypoints_{}.json'.format(dataDir, subDir, dataType)
coco_kps = COCO(annFile)
# load and display keypoints annotations
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
# plt.show()

# initialize COCO api for caption annotations
annFile = '{}/{}/annotations/captions_{}.json'.format(dataDir, subDir, dataType)
coco_caps = COCO(annFile)
# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(img_io)
plt.axis('off')
plt.show()