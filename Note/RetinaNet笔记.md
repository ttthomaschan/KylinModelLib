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

- RetinaNet：Anchor-based + FPN

- 类别不平衡（正负样本不平衡）：1）训练效率低，大量 Easy Negative 样本造成信息冗余，对模型性能提升没有用处；2）模型退化，Easy Negative 样本在训练过程中占主导位置。

- 鲁棒性评估：设计损失函数需要考虑鲁棒性，一般会减低离群点(outliers)的权重（e.g. Huber loss）。

  Focal Loss 的设计与常规设计相反，降低 Easy example(inliers)的权重，而聚焦在难样本上。



#### Focal Loss

- 目的：解决前景背景样本数量极度不平衡的问题（e.g. 1:1000）。
- 基础：Binary Cross Entropy（BCE）

$$
CE(p,y) = \begin{cases}
-log(p) & \text{if $$ y=1} \\ 
-log(1-p) & otherwise 
\end{cases}
$$

- 改进版：Balanced Cross Entropy 【引入参数alpha解决了**正负样本**不均衡，但没有解决**难易样本**不均衡】（原实验中alpha=0.25效果较好，相当于1:3）

$$
CE(p,y) = \begin{cases}
-\alpha log(p) & \text{if $$ y=1} \\ 
-(1-\alpha)log(1-p) & otherwise 
\end{cases}
$$

​		小结：平衡了正负样本的数量，但实际上，目标检测中难易样本比例也很不平衡。更重要的是，容易样本数量很多且损失很小，会主导了总的损失函数。

- Focal Loss 定义 【引入参数gamma，解决了难易样本不平衡】（原实验中 gamma=2 效果最好）

$$
FL(p,y) = \begin{cases}
-\alpha(1-p)^\gamma log(p) & \text{if $$ y=1} \\ 
-(1-\alpha)p^\gamma log(1-p) & otherwise 
\end{cases}
$$



#### RetinaNet Detector

![](https://gitee.com/jchencp/notepics/raw/master/retinanet_structure.png)

- Backone(ResNet) + Neck(FPN) + Head(classification + regression)

- Anchor-based:

  (1) ratios * sizes 

  (2) IOU threshold

- Inference and Training

  (1) 推理 

  (2) 初始化

  (3) 优化



#### Experiment





#### Conclusion



#### 进一步启发点

- 如何提高RetinaNet检测速度
- 是否有必要使用所有的Anchors





