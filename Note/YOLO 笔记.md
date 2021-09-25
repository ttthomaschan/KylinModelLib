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

YOLOv1 输入图像大小固定为 448x448x3，经过24个卷积层（7x7, 3x3, 1x1）提取特征后，特征图大小为 7x7x1024；然后是两个全连接层（4096, 30）。

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

​	共用SxSxB(eg. 7x7x2=98)个框，<u>**【具体实现】**</u>

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

![](https://i.loli.net/2021/09/25/hw9UkbOyXe5BRlI.png)