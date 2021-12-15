## RetinaNet

### 出发点

##### 目标检测中，一阶网路比二阶网络精度低，**归因于：样本不平衡。**

- 一张图片中，目标（前景）所占比例远远小于背景所占比例，因此，数量上：

  负样本 >> 正样本

  难样本 > 容易样本

- 难易样本 & 正负样本： Easy Positive / Hard Positive / Hard Negative / Easy Negative

##### 样本不均衡，导致网络性能不好的原因：

- 针对所用负样本而言，数量过多会造成整体Loss很大，从而主导损失函数，不利于收敛。（问题一）
- 针对单个负样本而言，单个Loss很小，反向计算时梯度小。因此，Easy Negative 对参数的收敛作用很有限。训练过程中更需要单体Loss大的对参数收敛影响大的样本，如 Hard Positive / Hard Negative 样本。（问题二）

##### 二阶网络 Faster-RCNN ：

- 在 RPN 中，根据前景分数过滤掉大量背景概率高的 Easy Negative 样本。（解决问题二）
- 会根据 IOU 来调整正负样本的比例，一般设置为1:3。（解决问题一）

##### 难负样本挖掘 Hard Negative Mining：

- 通过对 Loss 排序，选出 Loss 最大的样本（图像级别？锚框级别？）来进行训练，这样能保证训练的区域都是难样本。
- 存在缺陷：容易样本都去掉了，会造成 Easy Positive 样本无法进一步提升训练精度。

##### 本文提出了 Focal Loss，解决了 One-stage 目标检测网络中 正负/难易 样本比例失衡的问题。



#### Abstract

- 在目标检测任务中，二阶网络精度最高。

- 一阶网络，前景背景类别不平衡。

- 提出 Focal Loss， 调整标准交叉熵损失函数，降低容易样本的权重。

- 设计 RetinaNet， 验证 Focal Loss 的作用，发现 RetinaNet 在速度上大幅提升，并且在精度上匹配 SOTA 二阶网络。

  

#### Introduction

- 二阶网络：第一阶段生成稀疏的候选框（已初步过滤掉大量背景框）；第二阶段分类背景和目标细类，并回归。
- 二阶网络中，传统的候选框生成方法( e.g. Selective Search, EdgeBoxes, DeepMask, RPN ），可以快速过滤背景样本（10k -> 1~2k）。
- 一阶网络中，生成框的数量量级在100k。
- Focal Loss 是动态调节的交叉熵损失函数。
- RetinaNet 在精度上超越了二阶的 SOTA 模型。



#### Related Work

- 传统分类器（特征提取+传统分类器）：特征提取（HOG / DPMs / sliding-window）；分类器（SVM / LR）
- 二阶检测网络（特征提取+神经网络分类器）：特征提取（Selective Search / **RPN**）；分类器（**CNN**）
- 一阶检测网络：OverFeature (the first) / YOLO / SSD
- RetinaNet：Anchor-based / FPN



Focal Loss





RetinaNet Detector



Experiment



Conclusion





