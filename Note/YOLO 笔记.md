## YOLO 系列

### YOLO v1

#### 摘要

- 将检测问题看作回归问题；

- 检测和分类共用一个网络，一阶端到端网络；

- 背景误分类少，但检测定位精确度欠佳；

- 相较其他同期模型，泛化性能稍好，迁移能力强。

  

#### 1. 介绍

二阶网络：分类+检测

- DPM-- sliding window
- R-CNN -- region proposal

一阶网络：

- YOLO -- end to end

YOLO 优势：

- 速度快；
- 没有使用anchor，对全局信息（分类和外观信息）把握好；
- 学习到更加泛化的特征，迁移学习能力较强。



#### 2. 联合检测

- 先将原图划分成 S*S 个网格（grid），如果一个物体的中心的中心落在一个网格单元（grid cell）内，那么这个grid cell 就负责检测（分类+检测）这个物体。
- 每个grid cell预测 B 个 bounding boxes，包括location和confidence score；置信度公式（实际上是IOU得分）如下：

$$
Confidence = Pr(Object)*IOU_{pred}^{truth}
$$

$$
Pr(Object) =
        \begin{cases}
            1,  & \text{object in cell} \\
            0, & \text{no object in cell} \\
        \end{cases}
$$

- 每个bounding box有5个预测值，除了置信度，还有位置 (x,y,w,h)；(x,y)是中心点坐标，w,h是相对于整幅图像的预测值(边框的宽和高)。**中心坐标的（x,y）相对于对应的网格归一化到0-1之间，w,h用图像的width和height归一化到0-1之间。**【把xywh归一化，有利于加快收敛？】
- 每个grid cell还需预测类别概率 ，只预测一组，不考虑bounding boxes数量。
- 预测阶段，输出类别概率与置信度（此处为IoU）的乘积。

$$
Pr(Class_i|Object)*Pr(Object)*IOU_{pred}^{truth}=Pr(Class_i)*IOU_{pred}^{truth}
$$

<u>**注意：**</u>

- YOLOv1的输出层为全连接层，因此训练模型只支持固定的输入分辨率。
- 每个grid cell可以预测B个bbox，但是最终只选择IOU最高的一个，因此每个gird cell只能预测一个类别，导致一个严重的**局限：当一个grid cell中有多个类别的物体时，不能全部识别。**

##### 2.1 网络设计

![](https://gitee.com/jchencp/notepics/raw/master/YOLOv1_structure.png)

YOLOv1 输入图像大小固定为 448x448x3，经过24个卷积层（7x7, 3x3, 1x1）提取特征后，特征图大小为 7x7x1024；然后是两个全连接层（4096, 30）; 整个网络没有使用 BN 层，用了一层  Dropout 层。

##### 2.2 训练

- 激活函数选用 leaky reLU。
- 增加权重系数 -- location和classification损失权重相同不合理：有大量背景网格（负样本），容易把置信度推向0，会影响收敛速度。
- 对w,h使用开方处理 -- sum-squared error对不同大小物体的偏移量量纲存在问题：相同偏移量，对于大物体和小物体的影响程度相差很大。
- 训练参数：

 	epoches=135
 	
 	batch size=64
 	
 	optimizer: momentum=0.9, decay=0.0005
 	
 	lr schedule: 1e-3(first epoch) --> 1e-2(75 epoches) --> 1e-3(30 epoches) --> 1e-4(30 epoches) 

​	 dropout=0.5

**匹配策略：**

- 共有SxSxB(eg. 7x7x2=98)个框，<u>**【具体实现】**</u>

![](https://gitee.com/jchencp/notepics/raw/master/YOLOv1_grid.png)

- 如果一个物体的中心点落在了某一个网格中，那么这个网格就负责回归这个物体的位置。

举例：

​	如上图，假设左下角网格坐标为(1,1)，而小狗所在的最小包围框的中心落在网格(2,3)，那么在7x7个网格里，网格(2,3) 负责预测小狗，其他没有物体中心点落下的网格，则不负责预测任何物体。	

- *每个网格(grid cell) 的 B 个预测框初始尺寸如何确定？* (x,y)初始值等于网格中心点，(w,h)有B对，预先设定。
- 每个网格的B个预测框最后只选定置信度最高的矩形框作为输出。 -- **【YOLO v1最多预测S*S个物体】**
- (x,y) 是 BBox 的中心坐标，以中心相对于该网格左上角坐标的偏移值（如小狗网格(2,3)的偏移值），使x/y范围属于0到1。
- (w,h) 基于原图进行归一化（分别除以图像的w和h），同样使得w/h范围属于0到1。

**损失函数：**

<img src="https://gitee.com/jchencp/notepics/raw/master/YOLOv1_loss.png" style="zoom: 67%;" />

##### 2.3 推理

使用NMS进行后处理，提升 2-3% mAP (相较二阶网络性能提升少)。 **<u>【NMS原理及实现】</u>**

##### 2.4 局限性

- 对重叠物体检测效果差，原因：每个grid cell只能预测一个类别。
- 在测试时，当同一类物体出现不常见的长宽比等其他特殊情况，泛化能力差，原因：？
- 小物体检测定位效果差，原因：尺寸固定，也没有特征融合步骤。

#### 3. 比较

#### 4. 实验

#### 5. 实际应用

#### 6. 总结



### YOLO v2

gitee: d5b886f215e73e4fc82eb181f88e760d

<img src="https://gitee.com/jchencp/notepics/raw/master/YOLO_v1_v2.png" style="zoom: 67%;" />

#### 	摘要

- 使用一系列优化手段（trick），使 YOLO 更快、更健壮。
- 自适应输入尺寸：通过输入不同的尺寸，进行速度和精度的折中。
- 提出联合训练：针对 detection 数据集数量少的问题，用已有的 classification 数据集拓展目标检测系统。

#### 1. 介绍

YOLOv2 和 YOLO9000 算法内核相同， 区别是训练方式不同：

- YOLO v2 用 COCO 数据集训练，可以识别80个种类；
- YOLO 9000 用 COCO + ImageNet 数据集联合训练，可以识别9000多个类别（细粒度分类）。

#### 2. 更好

##### 2.1 Batch Normalization （批归一化）

检测网络中，BN 层成了标配。

YOLO v2 在每个卷积层加了 BN 层，并且去掉了 Dropout 层，mAP 提升了2% (VOC 2007) 。

##### 2.2 High Resolution Classifier （分类网络高分辨率预训练）

检测网络的 backbone 一般会在 ImageNet 上做预训练。

YOLO v1，预训练时输入大小为224x224；

YOLO v2，预训练时输入大小为448x448，在 ImageNet 上训练10个 epoches，再使用检测数据集 COCO 进行微调。 mAP 提升了4% (VOC 2007) 。

##### 2.3 Convolution with Anchor Boxes （ anchor box 替换全连接层）



##### 2.4 Dimension Clusters （聚类生成anchor的宽和高）

创新点！

Anchor Box 宽高无需人为设定，将训练数据集的矩形框全部抽取出来，用 kmeans 聚类得到先验框的宽高。聚类类别个数需要人为设定，当 k=5, mAP略微提升。

注意：聚类必须定义聚类点（矩形框宽高(w,h)）之间的聚类函数，文中使用如下函数：
$$
d(box,centroid) = 1 - IOU(box,centroid)
$$
若以（1 - IOU）为距离函数，是否也需要用到(x,y)信息？不需要。

##### 2.5 Direct location prediction （预测绝对位置）

$$
b_x = \sigma(t_x) + c_x
$$

$$
b_y = \sigma(t_y) + c_y
$$

$$
b_w = p_we^{tw}
$$

$$
b_h = p_he^{th}
$$



- 预测值为：

$$
t_x, t_y,t_w,t_h
$$

- 激活函数使用 sigmoid
- Cx 和 Cy 分别表示网格坐标

##### 2.6 Fine-Grained Features （细粒度特征）

在 YOLOv1中最终的特征图大小为 7x7（即grid cell尺寸）；而在 YOLOv2 中，因为输入尺寸从原来的224增加到416，最终特征图大小为 13x13。 

作者认为特征图经过卷积层从 26x26 到 13x13 的过程中，损失了很多细粒度特征，导致小尺寸物体识别效果不佳。

因此引入 passthrough 层，将 26x26x512 的特征图，变成 13x13x2048 的特征图：

![](https://gitee.com/jchencp/notepics/raw/master/YOLOv2_passthrough_.jpg)

<u>**【具体实现】**</u>：1）两刀切？2）在每个2x2区域中分别选择一个区域？【两种方法的区别？】

##### 2.7 Multi-Scale Training （多尺度训练）

YOLOv2 中只有卷积层和池化层，所以对于输入图像的尺寸没有限制，整个网络的降采样倍数是32。

<u>**【具体实现】：**</u>

每10个batch转换一下输入图像的尺寸，在{320, 352, ..., 608}中随机选择一种。迫使卷积核学习不同比例大小尺寸的特征。当输入尺寸达到 544x544 时，mAP (VOC 2007 + 2012)  已经超过同期的 Faster-RCNN 和 SSD。

#### 3. 更快

提出新的特征提取 backbone 网络 -- Darknet-19（对标 VGG-16）。

特点：参数比 VGG-16 更少。

<img src="https://gitee.com/jchencp/notepics/raw/master/darknet-19.png" style="zoom:80%;" />

#### 4. 更强

全文最大创新点 -- 分类和检测联合训练， YOLOv2 --> YOLO9000。

- 分类网络使用 Softmax 层，前提是类别互斥。
- YOLOv2 中目标检测不依赖于类别预测，二者同时独立进行。
- 分类数据集 ImageNet，物体类别有上万个；而物体检测数据集 COCO，只有80个类别。而且物体检测/分割的打标签成本比物体分类打标签成本要高很多。因此提出联合训练的方案。
- 当模型输入是检测数据集时，标注信息有类别、有位置信息，那么使用完整loss函数进行反向传播；当模型输入只包含分类信息时，只计算分类loss进行反向传播，其余部分为零。
- 先在检测数据集上训练一定的epoch，待预测框的loss基本稳定后，再联合分类数据集、检测数据集进行交替训练。

- 为了平衡分类、检测数据量，需要对 COCO 数据集进行了上采样。
- 使用WordTree，解决了 ImageNet 与 COCO 之间的类别问题。
- **在类别中，不对所有的类别进行 softmax 操作，而对同一层级的类别进行 softmax。**
- 在预测时，从树的根节点开始向下检索，每次选取预测分值最高的子节点，直到所有选择的节点预测分值连乘后小于某一阈值时停止。
- 在训练时，如果标签为人，那么只对人这个节点以及其所有的父节点进行loss计算，而其子节点，男人、女人、小孩等，不进行loss计算。

#### 5. 总结



### YOLO v3

#### 摘要

- YOLOv3 同时期的 SOTA 目标检测网络是 RetinaNet。

- 在 COCO 数据集上，Titan X 计算：

  ​	YOLO v3：57.9 AP50，51ms；

  ​	RetinaNet：57.5 AP50，198ms。

  

#### 1. 介绍

YOLOv3 相比前作只有较小改动，但性能取得了提升。



#### 2. 处理方法

##### 2.1 检测框预测

BBox 预测框计算方式和 YOLOv2 一样。

##### 2.2 分类预测

- 实际应用中，多标签和重叠标签的情况很多。

  Softmax 适用于单一标签；

  Logistic 适用于多标签；

- 分类损失用 BCE loss。

##### 2.3 多尺度预测

参考 FPN 思想，对不同特征层进行融合后，并做预测。

*YOLOv3 取3层特征层融合，然后在每个网格回归时选用3种尺寸的先验框，**共9种先验框**。*<u>【看代码验证】</u>

##### 2.4 特征提取

使用新的 backbone 网络 -- Darknet-53（参考了 ResNet 思想）。

Darknet-53 对标 ResNet-101/152，在 ImageNet 上准确率持平，但效率更高。



#### 3. 效果

同期 SOTA 模型为 RetinaNet。

YOLOv3 相比 YOLOv2， 小物体检测性能提升很多，但是仍然不及中大型物体。

AP50 指标性能好。

物体检测定位能力仍然是弱点。



#### 4. 无效尝试

- Anchor box x,y offset predictions.
- Linear x,y predictions instead of logistic.
- Focal loss.
- Dual IOU thresholds and truth assignment. （上下界限阈值）



#### 5. 总结

https://github.com/ultralytics/yolov3
https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
https://www.cnblogs.com/pprp/p/12432540.html
https://cloud.tencent.com/developer/article/1583429
https://www.jianshu.com/p/d13ae1055302
https://blog.csdn.net/leviopku/article/details/82660381



### YOLO v4

#### 摘要



#### 1.介绍



#### 2.相关工作

##### 2.1 目标检测模型



##### 2.2 Bag of Freebies



##### 2.3 Bag of Specials



#### 3.方法论

##### 3.1 结构选择

##### 3.2 BoF & BoS 选择

##### 3.3 其他增强手段

##### 3.4 YOLOv4 结构及细节



#### 4.实验

##### 4.1 设置

##### 4.2 分类器特征的影响

##### 4.3 检测器特征的影响

##### 4.4 骨干网络和预训练权重的影响

##### 4.5 批大小的影响



#### 5.实验结果及总结


### YOLOX

#### 摘要



#### 1.介绍

https://matpool.com/blog/6125a42a5a27e902e6a03b74/
https://zhuanlan.zhihu.com/p/411045300
https://blog.csdn.net/u011622208/article/details/119146813

#### 2.YOLOX <u>【基本结构 及 创新点】</u>

2.1 

**训练参数：**

- COCO-2017
- 300 epochs with 5 warmup epochs
- SGD &  linear LR( lrxBS/64)
- Batchsize: 128 with 8 GPU

**基准模型：**

- YOLOv3 (Darknet-53 + SPP)
- EMA weights updating (EMA 权值更新)
- Cosine lr schedule
- IoU-aware branch (IoU感知分支) 
- BCE Loss for cls and obj branch; IoU Loss for reg branch

**解耦头（Decoupled head）**

在目标检测中，分类与回归任务的冲突是一种常见问题。

采用解耦头替换YOLO的检测头可以显著改善模型收敛速度

**数据增强**

- Mosaic -- YOLOv3 中被提出，后广泛应用

- Mixup -- 最先被用于分类模型

  1）使用以上增强手段，但是**在最后15个epoch关闭使用**。

  2）**同时使用以上两个增强手段后，在 ImageNet 上的预训练就没有意义和效果了**。

**Anchor-Free**



#### 3.SOTA 模型比较



#### 4.总结


