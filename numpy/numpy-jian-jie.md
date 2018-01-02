> Source: [https://wolfsonliu.github.io/archive/python-xue-xi-bi-ji-numpy.html](https://wolfsonliu.github.io/archive/python-xue-xi-bi-ji-numpy.html)

这里只介绍在本书中用到的用法

### 创建数组

* np.array: 可以把 list, tuple, array, 或者其他的序列模式的数据转创建为 ndarray, 默认创建一个新的 ndarray.

```
data = np.array(range(5), dtype = int)
data
# array([0, 1, 2, 3, 4])
```

* np.zeros, np.zeros\_like: 创建元素全为 0 的数组, 类似 np.ones.

```
np.zeros(10)
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
```

#### 统计量

NumPy 中有简单的统计计算的方法或者函数, 有着很好的效率. 统计量计算的函数和方法可以通过更高层调用, 如`np.sum(arr)`, 或者使用数组的方法, 如`arr.sum`.

* `np.sum`: 计算数组的和, 可以设置参数 axis 为 0 或者 1 单独计算每行或每列的和.
* `np.mean`: 计算数组的均值, 可以设置参数 axis 为 0 或者 1 单独计算每行或每列的均值.
* `np.std`: 计算数组的标准差, 可以设置参数 axis 为 0 或者 1 单独计算每行或每列的标准差.
* `np.var`: 计算数组的标准差, 可以设置参数 axis 为 0 或者 1 单独计算每行或每列的标准差.
* `np.min`,`np.max`: 计算数组的最小值或最大值, 可以设置参数 axis 为 0 或者 1 单独计算每行或每列的最小值或最大值.
* `np.argmin`,`np.argmax`: 计算数组的最小值或最大值的 index, 可以设置参数 axis 为 0 或者 1 单独计算每行或每列的最小值或最大值 index.
* `np.cumsum`: 累加.
* `np.cumprod`: 累乘.

```
nd = np.random.randn(100)  # 随机产生 100 个随机数.
nd.sum()
# -13.300540142526414
np.mean(nd)
# -0.13300540142526412
nd.std()
# 1.0860199448302112
nd.var()
# 1.1794393205690152
nd.min(), np.max(nd)
# (-3.2346987329963834, 2.3950110710189687)
nd.argmin(), np.argmax(nd)
# (24, 31)
nd.cumsum()
# array([ -0.55436107,  -0.86184171,  -2.47687452,  -5.0649675 ,
#            ...            ...            ...          ...
#        -13.96853959, -13.59106636, -14.16799745, -13.30054014])
```

#### 线性代数

NumPy 中提供了一些线性代数运算的函数, 在 linalg 中有更全的线性代数的计算函数. 更多的线性代数相关函数包含在 scipy.linalg 包中.

* `np.dot`: 矩阵乘法.
* `np.transpose`: 返回转置, 也可以使用一个 ndarray 的 nd.T 属性.
* `np.diagonal`: 返回矩阵的对角元素.
* `np.trace`: 返回矩阵的迹.
* `np.linalg.eig`: 返回方阵的特征值和特征向量.
* `np.linalg.inv`: 返回方阵的逆矩阵.
* `np.linalg.pinv`: 返回方阵的 Moore-Penrose 伪逆矩阵.
* `np.linalg.solve`: 解线性方程组, 输入值是系数矩阵和线性方程组的常数项.
* `np.linalg.det`: 求方阵的行列式.
* `np.linalg.matrix_rank`: 求一个矩阵的秩.
* `np.linalg.svd`: 奇异值分解.



