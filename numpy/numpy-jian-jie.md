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



