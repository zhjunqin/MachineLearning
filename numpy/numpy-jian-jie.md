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

```
>>> a = np.array([[1,2],[3,4]]) 
>>> print(a)
[[1 2]
 [3 4]]
>>> b = np.array([[11,12,10],[13,14,9]]) 
>>> print(b)
[[11 12 10]
 [13 14  9]]
>>> np.dot(a,b)
array([[37, 40, 28],
       [85, 92, 66]])
>>> b.T
array([[11, 13],
       [12, 14],
       [10,  9]])
```

#### 数值计算

下面是一些单目运算函数.

* `np.max`: 返回 ndarray 中的最大值.
* `np.argmax`: 返回 ndarray 中最大的值的序号.
* `np.min`: 返回 ndarray 中的最小值.
* `np.argmin`: 返回 ndarray 中最小的值的序号.
* `np.absolute`: 计算绝对值.`np.absolute(a)`或者`np.abs(a)`, 对于非复数的数组,`np.fabs`速度更快.
* `np.exp`: 计算 e 的指数,`e ** x`.
* `np.sqrt`: 计算平方根,`x ** 0.5`.
* `np.square`: 计算平方,`x ** 2`.
* `np.log`,`np.log10`,`np.log2`,`np.log1p`: 分别为以 e, 10, 2 为底取 log, 和`log(1 + x)`.
* `np.sign`: 取数值的正负号.
* `np.ceil`: 计算比每一个元素大或相等的最小的整数.
* `np.floor`: 计算比每一个元素小或相等的最大的整数.
* `np.rint`: 近似到最近的整数.
* `np.clip`: 返回一个 ndarray, 其元素的值限制在给定的最大最小值之间. 如果原 ndarray 的值在给定的范围之外, 则替换成最大或最小值.
* `np.modf`: 返回一个 tuple, 包含两个数组, 一个是小数部分, 一个是整数部分.
* `np.cos`,`np.cosh`,`np.sin`,`np.sinh`,`np.tan`,`np.tanh`,`np.arccos`,`np.arccosh`,`np.arcsin`,`np.arcsinh`,`np.arctan`,`np.arctanh`: 三角函数和反三角函数.

```
nd = np.random.randn(10)
# array([-1.38153059, -0.66621482, -0.58001284, -0.81628342,  0.0656215 ,
#        -0.01538155, -0.77812592,  0.94664076,  0.85143997, -0.68542156])
np.absolute(nd)
# array([ 1.38153059,  0.66621482,  0.58001284,  0.81628342,  0.0656215 ,
#         0.01538155,  0.77812592,  0.94664076,  0.85143997,  0.68542156])
```

还有一些双目运算函数.

* `np.add`,`+`: 两个数组元素一一对应相加.
* `np.substract`,`-`: 两个数组元素一一对应相减.
* `np.multiply`,`*`: 两个数组元素一一对应相乘.
* `np.devide`,`/`: 两个数组元素一一对应相除.
* `np.floor_divide`,`np.remainder`,`np.mod`,`np.fmod`:`np.floor_divide`返回一一对应相除的最大整数商, 即 floor, 而`np.remainder`或`np.mod`则返回余数. 同时,`np.fmod`返回的余数则根据被除数和除数的符号可能是负数.
* `np.power`: 计算幂, 以第一个数组中元素为底, 以第二个数组中元素为指数.
* `np.maximum`,`np.fmax`: 一一比较两个数组中元素大小, 返回相应位置最大的.`np.fmax`会忽略 np.NAN, 而`np.maximum`
  则返回 np.NAN.
* `np.minimum`,`np.fmin`: 一一比较两个数组中元素大小, 返回相应位置最小的.`np.fmin`会忽略 np.NAN, 而`np.minimum`
  则返回 np.NAN.
* `np.copysign`: 把第二个数组中的元素的符号复制给第一个数组中的相应元素.

```
>>> a = np.array([1,2,3,4])
>>> b = np.array([1,0,-1,1])
>>> a*b
array([ 1,  0, -3,  4])
```



