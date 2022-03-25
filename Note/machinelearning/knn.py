# -*- coding:utf-8 -*-
"""
# @file name  : knn.py
# @author     : JLChen
# @date       : 2022-02
# @brief      : KNN demo
"""
'''
Note #1:
def KNeighborsClassifier(n_neighbors = 5,
                        weights='uniform',
                        algorithm = '',
                        leaf_size = '30',
                        p = 2,
                        metric = 'minkowski',
                        metric_params = None,
                        n_jobs = None)
:param										
- n_neighbors: KNN 中的"K"值。
- weights: 最普遍的 KNN 算法无论距离如何，权重都一样，也可以让距离更近的点让它更加重要。
    参数选项如下:
    • 'uniform': 不管远近权重都一样，就是最普通的 KNN 算法的形式。
    • 'distance': 权重和距离成反比，距离预测目标越近具有越高的权重。
    • 自定义函数: 自定义一个函数，根据输入的坐标值返回对应的权重，达到自定义权重的目的。
- algorithm: 在 sklearn 中，要构建 KNN 模型有三种构建方式:
    1）暴力法，就是直接计算距离存储比较；
    2）使用KD树构建；
    3）使用球树构建。 
    【 其中暴力法适合数据较小的方式，否则效率会比较低。如果数据量比较大一般会选择用 KD 树构建，而当 KD 树也比较慢的时候，则可以试试球树来构建。】
    参数选项如下:
	• 'brute': 蛮力实现
	• 'kd_tree': KD树实现 KNN
	• 'ball_tree': 球树实现 KNN 
	• 'auto'： 默认参数，自动选择合适的方法构建模型
- leaf_size: 如果是选择蛮力实现，那么这个值是可以忽略的；当使用KD树或球树，它就是是停止建子树的叶子节点数量的阈值。默认30，但如果数据量增多这个参数需要增大，否则速度过慢不说，还容易过拟合。
- p: 和metric结合使用的，当metric参数是"minkowski"的时候，p=1为曼哈顿距离， p=2为欧式距离。默认为p=2。
- metric: 指定距离度量方法，一般都是使用欧式距离。
	• 'euclidean' ：欧式距离
	• 'manhattan'：曼哈顿距离
	• 'chebyshev'：切比雪夫距离
	• 'minkowski'： 闵可夫斯基距离，默认参数
- n_jobs: 指定多少个CPU进行运算。默认是-1，也就是全部都算。
'''

'''
Note #2:
在使用KNN算法之前，我们要先决定K的值是多少，要选出最优的K值，可以使用sklearn中的交叉验证方法。
'''

### Step 1: 计算出合适的K值
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


iris = load_iris()
x = iris.data
y = iris.target

def findValueK():
    k_range = range(1, 31)
    k_error = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x, y, cv=6, scoring='accuracy')
        k_error.append(1-scores.mean())

    plt.plot(k_range, k_error)
    plt.xlabel("Value of K")
    plt.ylabel('Error')
    plt.show()

    return k_error.index(min(k_error)) + 1

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target


if __name__ == "__main__":
    n_neighbors = 11 # findValueK()

